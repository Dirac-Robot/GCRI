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
    Hypothesis,
    Strategies,
    FailureCategory,
    ActiveConstraints,
    RawHypothesis,
    AggregatedBranch,
    SandboxCurationResult,
    create_decision_schema,
)
from gcri.graphs.states import (
    TaskState, BranchState,
    HypothesisResult, IterationLog, StructuredMemory
)
from gcri.graphs.callbacks import GCRICallbacks, CLICallbacks
from gcri.graphs.aggregator import HypothesisAggregator
from gcri.graphs.generators import DefaultBranchesGenerator
from gcri.tools.cli import build_model, build_decision_model, BranchContainerRegistry, set_global_variables
from gcri.tools.utils import SandboxManager


class TaskAbortedError(Exception):
    """Raised when a task is aborted by the user."""
    pass


class GCRI:

    def __init__(self, config, schema=None, abort_event=None, callbacks=None):
        self.config = config
        set_global_variables()
        from gcri.tools.cli import GlobalVariables
        GlobalVariables.CONFIG = config
        self.schema = schema
        self.sandbox = SandboxManager(config)
        self.abort_event = abort_event
        self.callbacks = callbacks or CLICallbacks()

        with open(config.templates.global_rules, 'r') as f:
            self.global_rules = f.read()

        self.aggregator = HypothesisAggregator(config, self.sandbox.docker_sandbox)
        self._branches_generator = DefaultBranchesGenerator(
            config, self.sandbox, abort_event, self.global_rules, self.callbacks
        )

        # Build main graph
        graph = StateGraph(TaskState)

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

        # Phase 3: Decision (directly after aggregation — verification is per-branch)
        graph.add_node('decision', self.decide)
        graph.add_node('update_memory', self.update_memory)

        # Connect edges
        graph.add_edge(START, 'generate_branches')
        graph.add_edge('generate_branches', 'aggregate_hypotheses')
        graph.add_edge('aggregate_hypotheses', 'decision')
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
        self.callbacks.on_phase_change('aggregation', iteration=state.count)
        logger.bind(ui_event='phase_change', phase='aggregation').info(
            'Starting Hypothesis Aggregation...'
        )
        source_containers = {
            hyp.index: hyp.container_id for hyp in state.raw_hypotheses
        }

        # Use aggregator to combine/filter hypotheses (with intelligent file merging)
        aggregation_result = self.aggregator.aggregate(state, source_containers)

        # Clean up discarded branch containers
        for hyp in state.raw_hypotheses:
            if hyp.index in aggregation_result.discarded_indices:
                if hyp.container_id:
                    logger.debug(f'🗑️ Cleaning up discarded branch {hyp.index} container')
                    self.sandbox.docker_sandbox.clean_up_container(hyp.container_id)

        # Build aggregated_result directly from aggregated branches for decision
        targets = self.config.protocols.aggregate_targets
        aggregated_results = []
        for branch in aggregation_result.branches:
            result_dict = {
                'strategy': f'Aggregated-{branch.index}',
                'hypothesis': branch.combined_hypothesis,
                'counter_example': '',
                'adjustment': branch.merge_reasoning,
                'counter_strength': 'none',
                'index': branch.index,
                'source_indices': branch.source_indices,
                'produced_files': branch.produced_files,
            }
            aggregated_results.append({key: result_dict.get(key) for key in targets})

        logger.info(f'📊 Aggregation: {len(aggregation_result.branches)} branches forwarded to decision')
        return {
            'aggregated_branches': aggregation_result.branches,
            'aggregated_result': aggregated_results,
        }




    def generate_branches(self, state: TaskState):
        self._check_abort()
        self.callbacks.on_phase_change('strategy', iteration=state.count)
        logger.info(f'Iter #{state.count+1} | Generating branches via {type(self._branches_generator).__name__}...')
        result = self._branches_generator.generate(state)
        return result

    def _check_abort(self):
        if self.abort_event is not None and self.abort_event.is_set():
            logger.warning('🛑 Abort detected. Stopping execution.')
            raise TaskAbortedError('Task aborted by user.')

    def _load_template_with_rules(self, template_path, **format_kwargs):
        with open(template_path, 'r') as f:
            template = f.read().format(**format_kwargs)
        return f'{self.global_rules}\n\n{template}'

    def _invoke_with_retry(self, agent, template, error_context='agent'):
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

    def _run_verification(self, state, container_id, branch_index, strategy, reasoning, hypothesis, intent_analysis, source_info=None):
        self._check_abort()
        label = f'VerifyBranch[{branch_index}]' if source_info else f'Branch[{branch_index}]'
        logger.bind(
            ui_event='node_update', node='verification',
            branch=branch_index, data={'type': 'processing'}
        ).info(f'Iter #{state.count_in_branch+1} | {label} Verifying...')

        verification_config = self.config.agents.verification
        if branch_index < len(self.config.agents.branches):
            branch_cfg = self.config.agents.branches[branch_index]
            if hasattr(branch_cfg, 'verification') and branch_cfg.verification.get('model_id'):
                verification_config = branch_cfg.verification
        agent = build_model(
            verification_config.model_id,
            verification_config.get('gcri_options'),
            container_id=container_id,
            **verification_config.parameters
        ).with_structured_output(schema=Verification)

        template = self._load_template_with_rules(
            self.config.templates.verification,
            task=state.task_in_branch,
            strategy=strategy,
            reasoning=reasoning,
            hypothesis=hypothesis,
            intent_analysis=intent_analysis
        )
        verification = self._invoke_with_retry(agent, template, 'Verification agent')

        result = HypothesisResult(
            index=branch_index,
            strategy=strategy,
            reasoning=reasoning,
            hypothesis=hypothesis,
            counter_reasoning=verification.reasoning,
            counter_example=verification.counter_example,
            counter_strength=verification.counter_strength,
            adjustment=''  # Initialize empty, will be populated by refinement
        )

        log_data = {
            'counter_example': verification.counter_example,
            'counter_strength': verification.counter_strength,
            'container_id': container_id
        }
        if source_info:
            log_data['source_branches'] = source_info
        logger.bind(
            ui_event='node_update', node='verification',
            branch=branch_index, data=log_data
        ).info(
            f'Iter #{state.count_in_branch+1} | {label} Verification: '
            f'{verification.counter_strength.upper()} counter'
        )
        self.callbacks.on_verification_complete(
            state.count_in_branch, branch_index,
            verification.counter_strength, verification.counter_example
        )
        return {'results': [result], 'verify_count': state.verify_count+1}

    def verify(self, state: BranchState):
        return self._run_verification(
            state, state.container_id, state.index,
            state.strategy, state.reasoning, state.hypothesis,
            state.intent_analysis_in_branch
        )



    @classmethod
    def _get_failure_category_description(cls):
        return '\n'.join(f'- {code.value}' for code in FailureCategory)

    def decide(self, state: TaskState):
        self._check_abort()
        self.callbacks.on_phase_change('decision', iteration=state.count)
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
        evals = [e.model_dump(mode='json') if hasattr(e, 'model_dump') else e for e in getattr(decision, 'branch_evaluations', [])]
        self.callbacks.on_decision(
            state.count, decision.decision, decision.best_branch_index,
            decision.global_feedback, evals
        )
        return {
            'decision': decision.decision,
            'best_branch_index': decision.best_branch_index,
            'final_output': decision.final_output,
            'global_feedback': decision.global_feedback,
            'branch_evaluations': getattr(decision, 'branch_evaluations', [])
        }

    def _should_update_memory(self, state: TaskState) -> str:
        if state.decision:
            logger.debug('Decision=True, skipping memory update (no next iteration)')
            return END
        return 'update_memory'

    def update_memory(self, state: TaskState):
        self._check_abort()
        self.callbacks.on_phase_change('memory', iteration=state.count)
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

        try:
            for index in range(start_index, self.config.protocols.max_iterations):
                logger.info('=' * 60)
                self.callbacks.on_iteration_start(index, self.config.protocols.max_iterations)
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
                    self.callbacks.on_iteration_complete(index, result)
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
            self.callbacks.on_task_abort(e)
            _aborted = True
        except Exception as e:
            self.callbacks.on_task_error(e)
            _aborted = True
            raise
        else:
            _aborted = False
        finally:
            self.sandbox.clean_up()
            elapsed = time.time()-start_time
            if not _aborted:
                self.callbacks.on_task_complete(result, elapsed)
            logger.info(f'🧹 Sandbox clean-up completed.')
            logger.info(f'⏱️ Total elapsed time: {elapsed:.2f}s ({elapsed/60:.1f}min)')
        return result
