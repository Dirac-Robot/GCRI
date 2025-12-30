import json

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
    ActiveConstraints, create_decision_schema
)
from gcri.graphs.states import TaskState, BranchState, HypothesisResult, IterationLog, StructuredMemory
from gcri.graphs.callbacks import AutoCallbacks
from gcri.tools.cli import build_model, build_decision_model, BranchContainerRegistry
from gcri.tools.utils import SandboxManager


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
        self.schema = schema
        self.sandbox = SandboxManager(config)
        self.abort_event = abort_event
        self.callbacks = callbacks or AutoCallbacks()
        with open(config.templates.global_rules, 'r') as f:
            self.global_rules = f.read()

        graph = StateGraph(TaskState)
        branch = StateGraph(BranchState)

        branch.add_node('sample_hypothesis', self.sample_hypothesis)
        branch.add_node('reasoning_and_refine', self.reasoning_and_refine)
        branch.add_node('verify', self.verify)

        branch.add_edge(START, 'sample_hypothesis')
        branch.add_edge('sample_hypothesis', 'reasoning_and_refine')
        branch.add_edge('reasoning_and_refine', 'verify')
        branch.add_edge('verify', END)

        branch_workflow = branch.compile()
        graph.add_node('branch_executor', branch_workflow)

        strategy_generator_config = config.agents.strategy_generator
        strategy_agent = build_model(
            strategy_generator_config.model_id,
            strategy_generator_config.get('gcri_options'),
            **strategy_generator_config.parameters
        ).with_structured_output(schema=Strategies)

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

        self._strategy_agent = strategy_agent
        self._decision_agent = decision_agent
        self._memory_agent = memory_agent

        graph.add_node('sample_strategies', self.sample_strategies)
        graph.add_node('aggregate', self.aggregate)
        graph.add_node('decision', self.decide)
        graph.add_node('update_memory', self.update_memory)

        graph.add_edge(START, 'sample_strategies')
        graph.add_conditional_edges(
            'sample_strategies',
            self.map_branches,
            ['branch_executor']
        )
        graph.add_edge('branch_executor', 'aggregate')
        graph.add_edge('aggregate', 'decision')
        graph.add_edge('decision', 'update_memory')
        graph.add_edge('update_memory', END)

        self._graph = graph
        self._workflow = graph.compile()

    @property
    def graph(self):
        return self._graph

    @property
    def workflow(self):
        return self._workflow

    def map_branches(self, state: TaskState):
        logger.bind(ui_event='phase_change', phase='execution').info('Starting Branch Execution...')
        num_branches = min(len(self.config.agents.branches), len(state.strategies))
        sends = []

        for index in range(num_branches):
            container_id = self.sandbox.setup_branch(state.count, index)
            sends.append(
                Send(
                    'branch_executor',
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

    def aggregate(self, state: TaskState):
        aggregated_results = []
        targets = self.config.protocols.aggregate_targets
        for result in state.results:
            if result.counter_strength == 'strong' and not self.config.protocols.accept_all:
                continue
            result = result.model_dump(mode='json')
            converted_result = {key: result.get(key) for key in targets}
            aggregated_results.append(converted_result)
        return {'aggregated_result': aggregated_results}

    def sample_strategies(self, state: TaskState):
        logger.info(f'Iter #{state.count+1} | Request generating strategies...')
        locked_intent = state.intent_analysis or 'None (Analyze Fresh)'
        if state.intent_analysis:
            logger.info(f'Iter #{state.count+1} | Using LOCKED Intent: {locked_intent}')
        logger.bind(
            ui_event='node_update',
            node='strategy',
            data={'type': 'processing'}
        ).info('Generating strategies...')
        template = self._load_template_with_rules(
            self.config.templates.strategy_generator,
            task=state.task,
            feedback=state.feedback,
            num_hypothesis=len(self.config.agents.branches),
            locked_intent=locked_intent
        )
        strategies = self._invoke_with_retry(self.strategy_agent, template, 'Strategy agent')
        for index, strategy in enumerate(strategies.strategies):
            logger.info(f'Iter #{state.count+1} | Sampled strategy #{index+1}: {strategy}')
        current_intent = state.intent_analysis or strategies.intent_analysis or ''
        if not state.intent_analysis and strategies.intent_analysis:
            logger.info(f'Iter #{state.count+1} | Intent Locked: {current_intent}')
        logger.bind(
            ui_event='node_update',
            node='strategy',
            data={
                'task': state.task,
                'strategies': [s.model_dump() for s in strategies.strategies],
                'intent_analysis': current_intent,
                'strictness': strategies.strictness
            }
        ).info('Strategies generated.')
        return {
            'task_strictness': strategies.strictness,
            'strategies': strategies.strategies,
            'intent_analysis': current_intent
        }

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

    def sample_hypothesis(self, state: BranchState):
        """
        Generate an initial hypothesis for the given branch strategy.

        Uses the configured hypothesis agent to propose a solution approach
        based on the task, strategy, and accumulated feedback/memory.

        Args:
            state: BranchState containing task, strategy, and context.

        Returns:
            dict: Contains 'hypothesis' key with the generated hypothesis string.
        """
        self._check_abort()
        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=state.index,
            data={'type': 'processing'}
        ).info(f'Iter #{state.count_in_branch+1} | Sampling hypothesis for strategy #{state.index+1}...')
        container_id = state.container_id
        hypothesis_config = self.config.agents.branches[state.index].hypothesis
        agent = build_model(
            hypothesis_config.model_id,
            hypothesis_config.get('gcri_options'),
            container_id=container_id,
            **hypothesis_config.parameters
        ).with_structured_output(schema=Hypothesis)
        template = self._load_template_with_rules(
            self.config.templates.hypothesis,
            task=state.task_in_branch,
            strictness=state.strictness,
            strategy=state.strategy,
            intent_analysis=state.intent_analysis_in_branch
        )
        hypothesis = self._invoke_with_retry(agent, template, f'Hypothesis agent at strategy #{state.index+1}')
        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=state.index,
            data={'hypothesis': hypothesis.hypothesis, 'container_id': container_id}
        ).info(f'Iter #{state.count_in_branch+1} | Sampled hypothesis #{state.index+1}: {hypothesis.hypothesis}')
        return {'hypothesis': hypothesis.hypothesis}

    def reasoning_and_refine(self, state: BranchState):
        """
        Apply reasoning to refine the current hypothesis.

        Analyzes the hypothesis through structured reasoning and produces
        a refined version with supporting rationale.

        Args:
            state: BranchState with current hypothesis and context.

        Returns:
            dict: Contains 'reasoning' and refined 'hypothesis' keys.
        """
        self._check_abort()
        logger.bind(
            ui_event='node_update',
            node='reasoning',
            branch=state.index,
            data={'type': 'processing'}
        ).info(f'Iter #{state.count_in_branch+1} | Reasoning and refining hypothesis #{state.index+1}...')
        container_id = state.container_id
        reasoning_config = self.config.agents.branches[state.index].reasoning
        agent = build_model(
            reasoning_config.model_id,
            reasoning_config.get('gcri_options'),
            container_id=container_id,
            **reasoning_config.parameters
        ).with_structured_output(schema=Reasoning)
        template = self._load_template_with_rules(
            self.config.templates.reasoning,
            task=state.task_in_branch,
            strategy=state.strategy,
            hypothesis=state.hypothesis,
            intent_analysis=state.intent_analysis_in_branch
        )
        reasoning = self._invoke_with_retry(agent, template, f'Reasoning agent at hypothesis #{state.index+1}')
        logger.bind(
            ui_event='node_update',
            node='reasoning',
            branch=state.index,
            data={
                'reasoning': reasoning.reasoning,
                'hypothesis': reasoning.refined_hypothesis,
                'container_id': container_id
            }
        ).info(
            f'Iter #{state.count_in_branch+1} | '
            f'Refined hypothesis #{state.index+1}: {reasoning.refined_hypothesis}'
        )
        return {
            'reasoning': reasoning.reasoning,
            'hypothesis': reasoning.refined_hypothesis
        }

    @property
    def strategy_agent(self):
        return self._strategy_agent

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
        ).info(f'Iter #{state.count_in_branch+1} | Verifying refined hypothesis #{state.index+1}...')
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
            f'Iter #{state.count_in_branch+1} | '
            f'Counter-Example of Hypothesis #{state.index+1} (Counter Strength: {verification.counter_strength}): '
            f'{verification.counter_example}'
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
        is_last_run = (state.count+1 >= self.config.protocols.max_iterations)
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

    def update_memory(self, state: TaskState):
        """
        Update structured memory based on iteration results.

        Processes the current iteration's feedback and stores relevant
        learnings in memory for use in subsequent iterations.

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
        iteration_log = IterationLog(
            count_in_memory_log=state.count,
            branch_evaluations=state.branch_evaluations,
            global_feedback=state.global_feedback or ''
        )
        current_memory.history.append(iteration_log)
        global_feedback = state.global_feedback
        if global_feedback:
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

    def __call__(self, task, initial_memory=None, commit_mode='manual'):
        """Execute the GCRI reasoning loop for a given task."""
        self.sandbox.setup()
        result = None
        if isinstance(task, dict):
            memory, feedback, start_index = self._restore_from_state(task, initial_memory)
        else:
            memory = initial_memory if initial_memory is not None else StructuredMemory()
            feedback = ''
            start_index = 0
        try:
            for index in range(start_index, self.config.protocols.max_iterations):
                logger.bind(ui_event='phase_change', phase='strategy', iteration=index).info(
                    f'Starting Iteration {index+1}...'
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
                    logger.error(f'Iteration {index+1} error: {e}')
            else:
                logger.info('Final result is not deduced, but iteration count is over.')
        except (KeyboardInterrupt, TaskAbortedError) as e:
            is_abort = isinstance(e, TaskAbortedError)
            logger.warning('🛑 GCRI Task aborted by user.' if is_abort else 'GCRI Task interrupted by user (Ctrl+C).')
            if result:
                result['final_output'] = 'Task aborted by user.'
            else:
                result = {'final_output': 'Task aborted by user before first iteration completion.'}
        return result
