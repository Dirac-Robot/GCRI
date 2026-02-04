import json
import os
import time

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send
from loguru import logger
from pydantic import TypeAdapter

from gcri.graphs.schemas import (
    Verification,
    Reasoning,
    Hypothesis,
    Strategies,
    FailureCategory,
    ActiveConstraints,
    RawHypothesis,
    AggregatedBranch,
    SandboxCurationResult,
    create_decision_schema
)
from gcri.graphs.states import (
    TaskState, BranchState, VerificationBranchState,
    HypothesisResult, IterationLog, StructuredMemory
)
from gcri.graphs.callbacks import AutoCallbacks
from gcri.graphs.aggregator import HypothesisAggregator
from gcri.graphs.generators import get_branches_generator
from gcri.tools.cli import build_model, build_decision_model, BranchContainerRegistry, set_global_variables, set_external_memory
from gcri.tools.utils import SandboxManager
from gcri.memory.external_memory import ExternalMemory


class TaskAbortedError(Exception):
    """Raised when a task is aborted by the user."""
    pass


class GCRI:
    """
    Graph-based Collective Reasoning Interface.

    GCRI is a multi-branch reasoning system that generates hypotheses,
    refines them through reasoning, and verifies them against counterexamples.
    It uses a LangGraph-based workflow to orchestrate parallel branch execution
    and collective decision-making.

    Attributes:
        config: Configuration object containing model settings, templates, and protocols.
        schema: Optional Pydantic schema for structured output validation.
        sandbox: SandboxManager for isolated file system operations per branch.
        abort_event: Optional threading.Event to signal task abortion.
        callbacks: GCRICallbacks instance for environment-specific behavior (CLI, Web, etc.).
    """

    def __init__(self, config, schema=None, abort_event=None, callbacks=None):
        """
        Initialize GCRI with configuration and optional parameters.

        Args:
            config: Configuration object with model IDs, templates, and protocol settings.
            schema: Optional Pydantic BaseModel for structured final output.
            abort_event: Optional threading.Event for cooperative task cancellation.
            callbacks: Optional GCRICallbacks instance. Defaults to AutoCallbacks.
        """
        self.config = config
        set_global_variables()
        self.schema = schema
        self.sandbox = SandboxManager(config)
        self.abort_event = abort_event
        self.callbacks = callbacks or AutoCallbacks()
        with open(config.templates.global_rules, 'r') as f:
            self.global_rules = f.read()

        # Initialize aggregator and generator
        self.aggregator = HypothesisAggregator(config, self.sandbox.docker_sandbox)
        generator_type = getattr(config, 'branches_generator_type', 'default')
        self._branches_generator = get_branches_generator(
            generator_type, config, self.sandbox, abort_event, self.global_rules
        )
        logger.info(f'🔧 BranchesGenerator type: {generator_type} ({type(self._branches_generator).__name__})')

        # Initialize external memory if enabled
        self._external_memory = None
        if getattr(config, 'external_memory', {}).get('enabled', False):
            ext_path = config.external_memory.get('path')
            if not ext_path:
                ext_path = os.path.join(config.run_dir, 'external_memory.json')
            self._external_memory = ExternalMemory(ext_path)
            set_external_memory(self._external_memory)
            logger.info(f'🧠 External memory enabled: {ext_path}')

        # Build main graph
        graph = StateGraph(TaskState)

        # Verification branch workflow
        verification_branch = StateGraph(VerificationBranchState)
        verification_branch.add_node('verify', self.verify_aggregated)
        verification_branch.add_edge(START, 'verify')
        verification_branch.add_edge('verify', END)
        self._verification_workflow = verification_branch.compile()

        # Initialize decision and memory agents (core GCRI components)
        decision_config = config.agents.decision
        decision_schema = create_decision_schema(schema=schema)
        logger.info(f'🔧 Custom output schema applied: {decision_schema.__name__}')

        decision_agent = build_decision_model(
            decision_config.model_id,
            decision_config.get('gcri_options'),
            **decision_config.parameters
        ).with_structured_output(schema=decision_schema)

        memory_config = config.agents.memory
        memory_agent = build_model(
            memory_config.model_id,
            memory_config.get('gcri_options'),
            **memory_config.parameters
        ).with_structured_output(schema=ActiveConstraints)

        self._decision_agent = decision_agent
        self._memory_agent = memory_agent

        # Build main workflow graph
        # Phase 1: Hypothesis generation via BranchesGenerator
        graph.add_node('generate_branches', self.generate_branches)

        # Phase 2: Aggregation
        graph.add_node('aggregate_hypotheses', self.aggregate_hypotheses)

        # Phase 3: Verification
        graph.add_node('verification_executor', self._verification_workflow)
        graph.add_node('collect_verification', self.collect_verification)

        # Phase 4: Decision
        graph.add_node('decision', self.decide)
        graph.add_node('update_memory', self.update_memory)

        # Connect edges
        graph.add_edge(START, 'generate_branches')
        graph.add_edge('generate_branches', 'aggregate_hypotheses')
        graph.add_conditional_edges(
            'aggregate_hypotheses',
            self.map_verification_branches,
            ['verification_executor']
        )
        graph.add_edge('verification_executor', 'collect_verification')
        graph.add_edge('collect_verification', 'decision')
        graph.add_conditional_edges(
            'decision',
            self._should_update_memory,
            {'update_memory': 'update_memory', END: END}
        )
        graph.add_edge('update_memory', END)

        self._graph = graph
        self._workflow = graph.compile()
        logger.info('✅ GCRI Instance Initialized (BranchesGenerator Architecture)')

    @property
    def graph(self):
        return self._graph

    @property
    def workflow(self):
        return self._workflow

    def map_hypothesis_branches(self, state: TaskState):
        """Map strategies to parallel hypothesis generation branches."""
        logger.bind(ui_event='phase_change', phase='hypothesis_generation').info(
            'Starting Hypothesis Generation...'
        )
        num_branches = min(len(self.config.agents.branches), len(state.strategies))
        logger.info(f'🌿 Spawning {num_branches} hypothesis branches...')
        sends = []

        for index in range(num_branches):
            container_id = self.sandbox.setup_branch(state.count, index)
            sends.append(
                Send(
                    'hypothesis_executor',
                    {
                        'index': index,
                        'count_in_branch': state.count,
                        'task_in_branch': state.task,
                        'strictness': state.task_strictness,
                        'strategy': state.strategies[index],
                        'feedback': state.feedback,
                        'container_id': container_id,
                        'intent_analysis_in_branch': state.intent_analysis
                    }
                )
            )
        return sends

    def aggregate_hypotheses(self, state: TaskState):
        """Aggregate raw hypotheses using HypothesisAggregator."""
        logger.bind(ui_event='phase_change', phase='aggregation').info(
            'Starting Hypothesis Aggregation...'
        )

        # Build source container map
        source_containers = {}
        for hyp in state.raw_hypotheses:
            source_containers[hyp.index] = hyp.container_id

        # Use aggregator to combine/filter hypotheses (with intelligent file merging)
        aggregation_result = self.aggregator.aggregate(state, source_containers)

        # Extract verification container map from aggregated branches
        # Aggregator now handles container merging internally
        verification_containers = {
            branch.index: branch.container_id
            for branch in aggregation_result.branches
            if branch.container_id
        }

        # Clean up discarded branch containers
        for hyp in state.raw_hypotheses:
            if hyp.index in aggregation_result.discarded_indices:
                if hyp.container_id:
                    logger.debug(f'🗑️ Cleaning up discarded branch {hyp.index} container')
                    self.sandbox.docker_sandbox.clean_up_container(hyp.container_id)

        return {
            'aggregated_branches': aggregation_result.branches,
            'verification_container_map': verification_containers
        }

    def map_verification_branches(self, state: TaskState):
        """Map aggregated branches to parallel verification branches."""
        logger.bind(ui_event='phase_change', phase='verification').info(
            'Starting Verification...'
        )
        num_branches = len(state.aggregated_branches)
        logger.info(f'🔍 Spawning {num_branches} verification branches...')
        sends = []

        for branch in state.aggregated_branches:
            container_id = state.verification_container_map.get(branch.index)
            if not container_id:
                logger.warning(f'No container for verification branch {branch.index}, skipping')
                continue

            sends.append(
                Send(
                    'verification_executor',
                    {
                        'index': branch.index,
                        'count_in_branch': state.count,
                        'task_in_branch': state.task,
                        'strictness': state.task_strictness,
                        'aggregated_branch': branch,
                        'container_id': container_id,
                        'intent_analysis_in_branch': state.intent_analysis
                    }
                )
            )
        return sends

    def collect_verification(self, state: TaskState):
        """Collect verification results and prepare for decision."""
        aggregated_results = []
        targets = self.config.protocols.aggregate_targets
        accepted_count = 0
        rejected_count = 0

        for result in state.results:
            if result.counter_strength == 'strong' and not self.config.protocols.accept_all:
                rejected_count += 1
                continue
            accepted_count += 1
            result_dict = result.model_dump(mode='json')
            converted_result = {key: result_dict.get(key) for key in targets}
            aggregated_results.append(converted_result)

        logger.info(f'📊 Verification: {accepted_count} accepted, {rejected_count} rejected (strong counter)')
        return {'aggregated_result': aggregated_results}

    # Legacy method name for backward compatibility
    def map_branches(self, state: TaskState):
        """Legacy method - redirects to map_hypothesis_branches."""
        return self.map_hypothesis_branches(state)

    def aggregate(self, state: TaskState):
        """Legacy method - redirects to collect_verification."""
        return self.collect_verification(state)

    def generate_branches(self, state: TaskState):
        """Execute the configured BranchesGenerator to produce raw hypotheses.

        This is the main entry point for hypothesis generation. The generator
        is selected based on config.branches_generator_type.

        Args:
            state: TaskState with task and feedback.

        Returns:
            dict: Contains 'strategies', 'raw_hypotheses', and related metadata.
        """
        self._check_abort()
        logger.info(f'Iter #{state.count+1} | Generating branches via {type(self._branches_generator).__name__}...')
        return self._branches_generator.generate(state)

    def _check_abort(self):
        """Check if abort has been requested and raise TaskAbortedError if so."""
        if self.abort_event is not None and self.abort_event.is_set():
            logger.warning('🛑 Abort detected. Stopping execution.')
            raise TaskAbortedError('Task aborted by user.')

    def _load_template_with_rules(self, template_path, **format_kwargs):
        """Load template file, format with kwargs, and prepend global rules."""
        with open(template_path, 'r') as f:
            template = f.read().format(**format_kwargs)
        return f'{self.global_rules}\n\n{template}'

    def _invoke_with_retry(self, agent, template, error_context='agent'):
        """Invoke agent with retry logic up to max_tries_per_agent times."""
        for _ in range(self.config.protocols.max_tries_per_agent):
            result = agent.invoke(template)
            if result is not None:
                return result
        raise ValueError(
            f'{error_context} could not generate output '
            f'for {self.config.protocols.max_tries_per_agent} times.'
        )

    @property
    def decision_agent(self):
        return self._decision_agent

    @property
    def memory_agent(self):
        return self._memory_agent

    def verify(self, state: BranchState):
        """
        Verify the refined hypothesis by attempting to find counterexamples.

        The verification agent critically examines the hypothesis and attempts
        to construct counterexamples that would invalidate it. The strength of
        counterexamples determines whether the hypothesis is accepted.

        Args:
            state: BranchState with refined hypothesis to verify.

        Returns:
            dict: Contains 'results' list with HypothesisResult including
                  counterexample and strength assessment.
        """
        self._check_abort()
        logger.bind(
            ui_event='node_update',
            node='verification',
            branch=state.index,
            data={'type': 'processing'}
        ).info(f'Iter #{state.count_in_branch+1} | Branch[{state.index}] Verifying...')
        container_id = state.container_id
        verification_config = self.config.agents.branches[state.index].verification
        agent = build_model(
            verification_config.model_id,
            verification_config.get('gcri_options'),
            container_id=container_id,
            **verification_config.parameters
        ).with_structured_output(schema=Verification)
        template = self._load_template_with_rules(
            self.config.templates.verification,
            task=state.task_in_branch,
            strategy=state.strategy,
            reasoning=state.reasoning,
            hypothesis=state.hypothesis,
            intent_analysis=state.intent_analysis_in_branch
        )
        verification = self._invoke_with_retry(agent, template, 'Verification agent')
        result = HypothesisResult(
            index=state.index,
            strategy=state.strategy,
            reasoning=state.reasoning,
            hypothesis=state.hypothesis,
            counter_reasoning=verification.reasoning,
            counter_example=verification.counter_example,
            counter_strength=verification.counter_strength,
            adjustment=verification.adjustment
        )
        logger.bind(
            ui_event='node_update',
            node='verification',
            branch=state.index,
            data={
                'counter_example': verification.counter_example,
                'counter_strength': verification.counter_strength,
                'container_id': container_id
            }
        ).info(
            f'Iter #{state.count_in_branch+1} | Branch[{state.index}] Verification: '
            f'{verification.counter_strength.upper()} counter'
        )
        return {'results': [result]}

    def verify_aggregated(self, state: VerificationBranchState):
        """
        Verify an aggregated hypothesis by attempting to find counterexamples.

        Similar to verify() but works with VerificationBranchState which contains
        AggregatedBranch instead of Strategy.

        Args:
            state: VerificationBranchState with aggregated hypothesis to verify.

        Returns:
            dict: Contains 'results' list with HypothesisResult.
        """
        self._check_abort()
        logger.bind(
            ui_event='node_update',
            node='verification',
            branch=state.index,
            data={'type': 'processing'}
        ).info(f'Iter #{state.count_in_branch+1} | VerifyBranch[{state.index}] Verifying aggregated hypothesis...')

        container_id = state.container_id
        branch = state.aggregated_branch

        # Use first available verification config (or fallback to branch 0)
        branch_idx = min(state.index, len(self.config.agents.branches)-1)
        verification_config = self.config.agents.branches[branch_idx].verification

        agent = build_model(
            verification_config.model_id,
            verification_config.get('gcri_options'),
            container_id=container_id,
            **verification_config.parameters
        ).with_structured_output(schema=Verification)

        # Build a pseudo-Strategy for template compatibility
        from gcri.graphs.schemas import Strategy
        pseudo_strategy = Strategy(
            name=f'Aggregated-{state.index}',
            description=branch.merge_reasoning,
            feedback_reflection='Aggregated from multiple branches',
            hints=[f'Source branches: {branch.source_indices}']
        )

        template = self._load_template_with_rules(
            self.config.templates.verification,
            task=state.task_in_branch,
            strategy=pseudo_strategy,
            reasoning=branch.merge_reasoning,
            hypothesis=branch.combined_hypothesis,
            intent_analysis=state.intent_analysis_in_branch
        )
        verification = self._invoke_with_retry(agent, template, 'Verification agent')

        result = HypothesisResult(
            index=state.index,
            strategy=pseudo_strategy,
            reasoning=branch.merge_reasoning,
            hypothesis=branch.combined_hypothesis,
            counter_reasoning=verification.reasoning,
            counter_example=verification.counter_example,
            counter_strength=verification.counter_strength,
            adjustment=verification.adjustment
        )

        logger.bind(
            ui_event='node_update',
            node='verification',
            branch=state.index,
            data={
                'counter_example': verification.counter_example,
                'counter_strength': verification.counter_strength,
                'container_id': container_id,
                'source_branches': branch.source_indices
            }
        ).info(
            f'Iter #{state.count_in_branch+1} | VerifyBranch[{state.index}] Verification: '
            f'{verification.counter_strength.upper()} counter'
        )
        return {'results': [result]}

    @classmethod
    def _get_failure_category_description(cls):
        descriptions = []
        for code in FailureCategory:
            descriptions.append(f'- {code.value}')
        return '\n'.join(descriptions)

    def decide(self, state: TaskState):
        """
        Make a collective decision based on all branch results.

        Evaluates all hypothesis results from parallel branches and determines:
        - Whether to accept one of the hypotheses as the final answer
        - Which branch produced the best result
        - What feedback to provide for the next iteration if rejected

        Args:
            state: TaskState with aggregated results from all branches.

        Returns:
            dict: Decision outcome including 'decision' boolean, 'best_branch_index',
                  'final_output', 'feedback', and updated 'memory'.
        """
        self._check_abort()
        logger.bind(ui_event='phase_change', phase='decision').info('Starting Decision Phase...')
        logger.info(f'Iter #{state.count+1} | Request generating final decision for current loop...')
        BranchContainerRegistry.set_containers(state.count, self.sandbox._branch_containers)
        file_contexts = self.sandbox.get_branch_context(state.count, len(state.results))
        template_path = self.config.templates.decision
        with open(template_path, 'r') as f:
            template = f.read()
        force_output = self.config.protocols.get('force_output', False)
        max_iter = self.config.protocols.max_iterations
        if not isinstance(max_iter, int):
            max_iter = int(max_iter) if max_iter is not None else 5
        is_last_run = (state.count+1 >= max_iter)
        if force_output and is_last_run:
            logger.warning(
                f'🚨 Force Output Triggered at Iter #{state.count+1}. '
                'Instructing agent to make a FINAL decision regardless of imperfections.'
            )
            force_instruction = (
                '\n\n'
                '!!! CRITICAL SYSTEM OVERRIDE - FINAL ITERATION !!!\n'
                '1. You have reached the MAXIMUM iteration limit.\n'
                '2. You MUST set \'decision\' to true.\n'
                '3. You MUST select the single best available branch via \'best_branch_index\', '
                'even if it is not perfect or has minor issues.\n'
                '4. Do NOT output \'decision\': false. Do NOT provide global_feedback.'
            )
            template += force_instruction
        if state.aggregated_result:
            aggregated_result = json.dumps(state.aggregated_result, indent=4, ensure_ascii=False)
        else:
            aggregated_result = None
        if self.schema:
            try:
                schema_json = json.dumps(self.schema.model_json_schema(), indent=2, ensure_ascii=False)
            except AttributeError:
                schema_json = str(self.schema)
            schema_desc = (
                f'MUST follow the specific JSON schema for "final_output" provided below:\n'
                f'{schema_json}\n'
                'Ensure ALL required fields (e.g., answer, confidence) are populated exactly as defined.'
            )
        else:
            schema_desc = 'String (only if True). The final adopted perfect answer.'
        template = template.format(
            task=state.task,
            aggregated_result=aggregated_result,
            file_contexts=file_contexts,
            failure_category_list=self._get_failure_category_description(),
            schema_desc=schema_desc,
            intent_analysis=state.intent_analysis
        )
        template = f'{self.global_rules}\n\n{template}'
        self.decision_agent.container_id = None  # Decision agent doesn't need container
        decision = self._invoke_with_retry(self.decision_agent, template, 'Decision agent')
        logger.bind(
            ui_event='node_update',
            node='decision',
            data=decision.model_dump()
        ).info(f'Decision: {decision.decision}')
        if decision.decision:
            if decision.best_branch_index >= 0:
                logger.info(f'Selected Best Branch Index: {decision.best_branch_index+1}')
            else:
                logger.warning('Decision is True but best_branch_index is -1')
        else:
            logger.info(f'Feedback: {decision.global_feedback}')
        return {
            'decision': decision.decision,
            'best_branch_index': decision.best_branch_index,
            'final_output': decision.final_output,
            'global_feedback': decision.global_feedback,
            'branch_evaluations': getattr(decision, 'branch_evaluations', [])
        }

    def _should_update_memory(self, state: TaskState) -> str:
        """Route to update_memory only if decision is False (continuing to next iteration)."""
        if state.decision:
            logger.debug('Decision=True, skipping memory update (no next iteration)')
            return END
        return 'update_memory'

    def update_memory(self, state: TaskState):
        """
        Update structured memory based on iteration results.

        Processes the current iteration's feedback and stores relevant
        learnings in memory for use in subsequent iterations.
        Also performs sandbox curation for cross-iteration artifact preservation.

        Args:
            state: TaskState with decision feedback and current memory.

        Returns:
            dict: Updated 'memory' and 'feedback' for next iteration.
        """
        self._check_abort()
        logger.bind(ui_event='phase_change', phase='memory').info('Updating Memory...')
        current_memory = state.memory
        memory_template_path = self.config.templates.memory
        with open(memory_template_path, 'r') as f:
            memory_template = f.read()

        # Collect sandbox file summary for this iteration
        branch_files = self.sandbox.get_branch_files(state.count)
        sandbox_file_summary = {
            idx: list(files.keys()) for idx, files in branch_files.items()
        }

        iteration_log = IterationLog(
            count_in_memory_log=state.count,
            branch_evaluations=state.branch_evaluations,
            global_feedback=state.global_feedback or '',
            sandbox_file_summary=sandbox_file_summary
        )
        current_memory.history.append(iteration_log)

        global_feedback = state.global_feedback
        if global_feedback:
            # Extract active constraints
            active_memory_template_path = self.config.templates.active_memory
            with open(active_memory_template_path, 'r') as f:
                active_memory_template = f.read()
            try:
                active_memory_template = active_memory_template.format(global_feedback=global_feedback)
                active_memory_template = f'{self.global_rules}\n\n{active_memory_template}'
                memory_agent = self.memory_agent
                active_memory = memory_agent.invoke(active_memory_template)
                new_constraints = active_memory.new_active_constraints
            except Exception as e:
                logger.error(
                    f'Iter #{state.count+1} | '
                    f'Constraint extraction failed: {e}. '
                    f'Falling back to old constraints.'
                )
                new_constraints = []
            current_set = set(current_memory.active_constraints)
            current_set.update(new_constraints)
            current_memory.active_constraints = list(current_set)

        # Perform sandbox curation for next iteration (independent of global_feedback)
        if branch_files:
            current_memory = self._curate_sandbox(
                state, current_memory, branch_files
            )

        integrated_feedback = current_memory.format_for_strategy(memory_template)
        logger.bind(
            ui_event='node_update',
            node='memory',
            data=current_memory.active_constraints
        ).info(f'Iter #{state.count+1} | Memory saved:\n{current_memory}')
        logger.info(f'Iter #{state.count+1} | Integrated feedback from memorized information:\n{integrated_feedback}')
        return {
            'memory': current_memory,
            'feedback': integrated_feedback
        }

    def _curate_sandbox(self, state: TaskState, memory: StructuredMemory, branch_files: dict) -> StructuredMemory:
        """
        Curate sandbox artifacts for next iteration.

        Args:
            state: Current TaskState with branch evaluations.
            memory: StructuredMemory to update.
            branch_files: Dict of branch_index -> {file_path: content}.

        Returns:
            Updated StructuredMemory with base_sandbox_container_id.
        """
        logger.info(f'🔍 Curating sandbox from {len(branch_files)} branches...')

        # Build the curation prompt
        sandbox_curator_path = self.config.templates.sandbox_curator
        with open(sandbox_curator_path, 'r') as f:
            curator_template = f.read()

        # Format branch evaluations
        branch_eval_str = '\n'.join([
            f'  - Branch {b.branch_index}: {b.status.value} '
            f'({b.failure_category.value if b.failure_category else "none"}) - {b.summary_hypothesis}'
            for b in state.branch_evaluations
        ])

        # Format branch files summary
        branch_files_str = '\n'.join([
            f'  - Branch {idx}: {len(files)} files - {list(files.keys())[:5]}'
            for idx, files in branch_files.items()
        ])

        curator_template = curator_template.format(
            task=state.task,
            decision=state.decision,
            global_feedback=state.global_feedback or 'None',
            branch_evaluations=branch_eval_str,
            branch_files=branch_files_str
        )
        curator_template = f'{self.global_rules}\n\n{curator_template}'

        # Build curator agent with SandboxCurationResult schema
        memory_config = self.config.agents.memory
        curator_agent = build_model(
            memory_config.model_id,
            memory_config.get('gcri_options'),
            **memory_config.parameters
        ).with_structured_output(schema=SandboxCurationResult)

        try:
            curation_result = curator_agent.invoke(curator_template)
            logger.info(
                f'📦 Sandbox curation: selected branches {curation_result.selected_branch_indices}, '
                f'reasoning: {curation_result.merge_reasoning[:100]}...'
            )

            if curation_result.selected_branch_indices:
                # Create base sandbox from selected branches
                base_container, summary = self.sandbox.create_base_sandbox(
                    curation_result.selected_branch_indices, state.count
                )
                if base_container:
                    memory.base_sandbox_container_id = base_container
                    memory.base_sandbox_summary = summary
                    logger.info(f'✅ Base sandbox created: {base_container[:12]}')

                    # Add curation feedback to memory
                    if curation_result.feedback_for_next_iteration:
                        memory.active_constraints.append(
                            f'[Sandbox] {curation_result.feedback_for_next_iteration}'
                        )
            else:
                logger.info('📭 No branches selected for sandbox curation')

        except Exception as e:
            logger.warning(f'Sandbox curation failed: {e}. Continuing without base sandbox.')

        return memory

    def _restore_from_state(self, task_dict, initial_memory):
        """Restore execution state from a previous run."""
        memory = initial_memory if initial_memory is not None else StructuredMemory()
        logger.info('🔄 State object detected. Resuming from previous state in memory...')
        try:
            task_content = task_dict.get('task', '')
            if 'memory' in task_dict:
                memory = TypeAdapter(StructuredMemory).validate_python(task_dict['memory'])
            feedback = task_dict.get('feedback', '')
            start_index = task_dict.get('count', -1)+1
            logger.info(f'Task: {task_content[:50]}...')
            logger.info(f'Resuming loop from index: {start_index}')
            return memory, feedback, start_index
        except Exception as e:
            logger.error(f'Failed to restore state from object: {e}')
            raise ValueError('Invalid state object provided.')

    def _handle_iteration_success(self, result, index, commit_mode):
        """Handle successful decision and optional commit."""
        logger.info('Final result is successfully deduced.')
        logger.info(f'Task Completed. Check sandbox: {self.sandbox.work_dir}')
        best_branch_index = result.get('best_branch_index', -1)
        if best_branch_index < 0:
            logger.warning('Decision is True but no branch index provided. Cannot commit automatically.')
            return True
        winning_branch_path = self.sandbox.get_winning_branch_path(index, best_branch_index)
        logger.info(f'🏆 Winning Branch Identified: Branch #{best_branch_index+1}')
        logger.info(f'📂 Location: {winning_branch_path}')

        # Update external memory on success (before commit)
        if self._external_memory:
            self._update_external_memory_on_success(result)

        commit_context = {
            'winning_branch_path': winning_branch_path,
            'best_branch_index': best_branch_index,
            'final_output': result.get('final_output')
        }
        should_commit = (
            commit_mode == 'auto-accept' or
            (commit_mode == 'manual' and self.callbacks.on_commit_request(commit_context))
        )
        if should_commit:
            self.sandbox.commit_winning_branch(winning_branch_path)
        else:
            logger.info('Changes discarded.')
        return True

    def _update_external_memory_on_success(self, result):
        """Update external memory on successful task completion (decision=True)."""
        try:
            ext_memory_template_path = self.config.templates.get('external_memory_update')
            if not ext_memory_template_path:
                logger.debug('external_memory_update template not configured, skipping')
                return
            with open(ext_memory_template_path, 'r') as f:
                ext_memory_template = f.read()
            result_summary = result.get('final_output', str(result))
            prompt = ext_memory_template.format(result_summary=result_summary)
            prompt = f'{self.global_rules}\n\n{prompt}'
            # Use memory agent to potentially save learnings
            memory_agent = self.memory_agent
            memory_agent.invoke(prompt)
            logger.info('External memory update completed on success')
        except Exception as e:
            logger.warning(f'External memory update on success failed: {e}')


    def __call__(self, task, initial_memory=None, commit_mode='manual'):
        """Execute the GCRI reasoning loop for a given task."""
        start_time = time.time()
        valid_modes = {'manual', 'auto-accept'}
        if commit_mode not in valid_modes:
            raise ValueError(
                f"Invalid commit_mode='{commit_mode}'. Must be one of: {valid_modes}"
            )
        self.sandbox.setup()
        result = None
        if isinstance(task, dict):
            memory, feedback, start_index = self._restore_from_state(task, initial_memory)
        else:
            memory = initial_memory if initial_memory is not None else StructuredMemory()
            feedback = ''
            start_index = 0
            # Load external memory rules if available
            if self._external_memory:
                ext_rules = self._external_memory.load(domain=getattr(self.config, 'task_domain', None))
                if ext_rules:
                    memory.active_constraints.extend(ext_rules)
                    logger.info(f'🧠 Loaded {len(ext_rules)} rules from external memory')
        try:
            for index in range(start_index, self.config.protocols.max_iterations):
                logger.info('=' * 60)
                logger.bind(ui_event='phase_change', phase='strategy', iteration=index).info(
                    f'🔄 Starting Iteration {index+1}/{self.config.protocols.max_iterations}'
                )
                try:
                    result = self.workflow.invoke({
                        'count': index,
                        'task': task,
                        'feedback': feedback,
                        'memory': memory
                    })
                    result = TypeAdapter(TaskState).validate_python(result).model_dump(mode='json')
                    self.sandbox.save_iteration_log(index, result)
                    if result['decision']:
                        self._handle_iteration_success(result, index, commit_mode)
                        # Save useful constraints to external memory on success
                        if self._external_memory and memory.active_constraints:
                            self._external_memory.save(
                                memory.active_constraints,
                                domain=getattr(self.config, 'task_domain', None)
                            )
                        break
                    else:
                        memory = TypeAdapter(StructuredMemory).validate_python(result['memory'])
                        feedback = result['feedback']
                except KeyboardInterrupt:
                    logger.warning(f'Iteration {index+1} interrupted by user. Stopping...')
                    raise
                except TaskAbortedError:
                    logger.warning(f'Iteration {index+1} aborted by user.')
                    raise
                except Exception as e:
                    import traceback
                    logger.error(f'Iteration {index+1} error: {e}\n{traceback.format_exc()}')
            else:
                logger.warning('⚠️ Max iterations reached without final decision.')
        except (KeyboardInterrupt, TaskAbortedError) as e:
            is_abort = isinstance(e, TaskAbortedError)
            logger.warning('🛑 GCRI Task aborted by user.' if is_abort else 'GCRI Task interrupted by user (Ctrl+C).')
            if result:
                result['final_output'] = 'Task aborted by user.'
            else:
                result = {'final_output': 'Task aborted by user before first iteration completion.'}
        finally:
            self.sandbox.clean_up()
            elapsed = time.time() - start_time
            logger.info(f'🧹 Sandbox clean-up completed.')
            logger.info(f'⏱️ Total elapsed time: {elapsed:.2f}s ({elapsed/60:.1f}min)')
        return result
