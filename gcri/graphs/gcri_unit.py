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
    DecisionProtoType,
    FailureCategory,
    ActiveConstraints, create_decision_schema
)
from gcri.graphs.states import TaskState, BranchState, HypothesisResult, IterationLog, StructuredMemory
from gcri.tools.cli import build_model, get_input
from gcri.tools.utils import SandboxManager


class GCRI:
    def __init__(self, config, schema=None):
        self.config = config
        self.schema = schema
        self.sandbox = SandboxManager(config)

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

        if schema:
            decision_schema = create_decision_schema(schema=schema)
            logger.info(f'🔧 Custom output schema applied: {decision_schema.__name__}')
        else:
            decision_schema = DecisionProtoType

        decision_agent = build_model(
            decision_config.model_id,
            decision_config.get('gcri_options'),
            work_dir=None,
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
        num_branches = min(len(self.config.agents.branches), len(state.strategies))
        sends = []

        for index in range(num_branches):
            branch_workspace = self.sandbox.setup_branch(state.count, index)
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
                        'work_dir': branch_workspace
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
        template_path = self.config.templates.strategy_generator
        with open(template_path, 'r') as f:
            template = f.read().format(
                task=state.task,
                feedback=state.feedback,
                num_hypothesis=len(self.config.agents.branches)
            )
        for _ in range(self.config.protocols.max_tries_per_agent):
            strategies = self.strategy_agent.invoke(template)
            if strategies is not None:
                break
        else:
            raise ValueError(
                f'Agent could not generate strategies '
                f'for {self.config.protocols.max_tries_per_agent} times.'
            )
        for index, strategy in enumerate(strategies.strategies):
            logger.info(f'Iter #{state.count+1} | Sampled strategy #{index+1}: {strategy}')
        return dict(task_strictness=strategies.strictness, strategies=strategies.strategies)

    def sample_hypothesis(self, state: BranchState):
        logger.info(f'Iter #{state.count_in_branch+1} | Request sampling hypothesis for strategy #{state.index+1}...')
        work_dir = state.work_dir
        hypothesis_config = self.config.agents.branches[state.index].hypothesis
        agent = build_model(
            hypothesis_config.model_id,
            hypothesis_config.get('gcri_options'),
            work_dir=work_dir,
            **hypothesis_config.parameters
        )
        template_path = self.config.templates.hypothesis
        with open(template_path, 'r') as f:
            template = f.read().format(
                task=state.task_in_branch,
                strictness=state.strictness,
                strategy=state.strategy
            )
        for _ in range(self.config.protocols.max_tries_per_agent):
            hypothesis = agent.with_structured_output(schema=Hypothesis).invoke(template)
            if hypothesis is not None:
                break
        else:
            raise ValueError(
                f'Agent could not generate hypothesis '
                f'for {self.config.protocols.max_tries_per_agent} times '
                f'at strategy #{state.index+1}.'
            )
        logger.info(f'Iter #{state.count_in_branch+1} | Sampled hypothesis #{state.index+1}: {hypothesis.hypothesis}')
        return dict(hypothesis=hypothesis.hypothesis)

    def reasoning_and_refine(self, state: BranchState):
        logger.info(f'Iter #{state.count_in_branch+1} | Request reasoning and refining hypothesis #{state.index+1}...')
        work_dir = state.work_dir
        reasoning_config = self.config.agents.branches[state.index].reasoning
        agent = build_model(
            reasoning_config.model_id,
            reasoning_config.get('gcri_options'),
            work_dir=work_dir,
            **reasoning_config.parameters
        )
        template_path = self.config.templates.reasoning
        with open(template_path, 'r') as f:
            template = f.read().format(
                task=state.task_in_branch,
                strategy=state.strategy,
                hypothesis=state.hypothesis
            )
        for _ in range(self.config.protocols.max_tries_per_agent):
            reasoning = agent.with_structured_output(schema=Reasoning).invoke(template)
            if reasoning is not None:
                break
        else:
            raise ValueError(
                f'Agent could not generate refined hypothesis '
                f'for {self.config.protocols.max_tries_per_agent} times '
                f'at hypothesis #{state.index+1}.'
            )
        logger.info(
            f'Iter #{state.count_in_branch+1} | '
            f'Refined hypothesis #{state.index+1}: {reasoning.refined_hypothesis}'
        )
        return dict(
            reasoning=reasoning.reasoning,
            hypothesis=reasoning.refined_hypothesis
        )

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
        logger.info(f'Iter #{state.count_in_branch+1} | Request verifying refined hypothesis #{state.index+1}...')
        work_dir = state.work_dir
        verification_config = self.config.agents.branches[state.index].verification
        agent = build_model(
            verification_config.model_id,
            verification_config.get('gcri_options'),
            work_dir=work_dir,
            **verification_config.parameters
        )
        template_path = self.config.templates.verification
        with open(template_path, 'r') as f:
            template = f.read()
        template = template.format(
            task=state.task_in_branch,
            strategy=state.strategy,
            reasoning=state.reasoning,
            hypothesis=state.hypothesis
        )
        for _ in range(self.config.protocols.max_tries_per_agent):
            verification = agent.with_structured_output(schema=Verification).invoke(template)
            if verification is not None:
                break
        else:
            raise ValueError(
                f'Agent could not generate verification '
                f'for {self.config.protocols.max_tries_per_agent} times.'
            )
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
        logger.info(
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
        logger.info(f'Iter #{state.count+1} | Request generating final decision for current loop...')
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
            schema_desc=schema_desc
        )
        self.decision_agent.work_dir = self.sandbox.work_dir
        for _ in range(self.config.protocols.max_tries_per_agent):
            decision = self.decision_agent.invoke(template)
            if decision is not None:
                break
        else:
            raise ValueError(
                f'Agent could not generate decision '
                f'for {self.config.protocols.max_tries_per_agent} times.'
            )
        logger.info(f'Decision: {decision.decision}')
        if decision.decision:
            logger.info(f'Selected Best Branch Index: {decision.best_branch_index+1}')
        else:
            logger.info(f'Feedback: {decision.global_feedback}')
        return {
            'decision': decision.decision,
            'best_branch_index': decision.best_branch_index,
            'final_output': decision.final_output,
            'global_feedback': decision.global_feedback,
            'branch_evaluations': decision.branch_evaluations
        }

    def update_memory(self, state: TaskState):
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
        logger.info(f'Iter #{state.count+1} | Memory saved:\n{current_memory}')
        logger.info(f'Iter #{state.count+1} | Integrated feedback from memorized information:\n{integrated_feedback}')
        return {
            'memory': current_memory,
            'feedback': integrated_feedback
        }

    def __call__(self, task, initial_memory=None, commit_mode='manual'):
        self.sandbox.setup()
        feedback = ''
        memory = initial_memory if initial_memory is not None else StructuredMemory()
        result = None
        if isinstance(task, dict):
            logger.info('🔄 State object detected. Resuming from previous state in memory...')
            try:
                task_content = task.get('task', '')
                if 'memory' in task:
                    memory = TypeAdapter(StructuredMemory).validate_python(task['memory'])
                feedback = task.get('feedback', '')
                start_index = task.get('count', -1)+1
                logger.info(f'Task: {task_content[:50]}...')
                logger.info(f'Resuming loop from index: {start_index}')
            except Exception as e:
                logger.error(f'Failed to restore state from object: {e}')
                raise ValueError('Invalid state object provided.')
        else:
            start_index = 0
        try:
            for index in range(start_index, self.config.protocols.max_iterations):
                logger.info(f'Starting Iteration {index+1}...')
                try:
                    result = self.workflow.invoke(
                        {
                            'count': index,
                            'task': task,
                            'feedback': feedback,
                            'memory': memory
                        }
                    )
                    result = TypeAdapter(TaskState).validate_python(result).model_dump(mode='json')
                    self.sandbox.save_iteration_log(index, result)
                    if result['decision']:
                        logger.info('Final result is successfully deduced.')
                        logger.info(f'Task Completed. Check sandbox: {self.sandbox.work_dir}')
                        best_branch_index = result.get('best_branch_index')
                        if best_branch_index is None:
                            logger.warning(
                                'Decision is True but no branch index provided. Cannot commit automatically.'
                            )
                            break
                        winning_branch_path = self.sandbox.get_winning_branch_path(index, best_branch_index)
                        logger.info(f'🏆 Winning Branch Identified: Branch #{best_branch_index+1}')
                        logger.info(f'📂 Location: {winning_branch_path}')
                        match commit_mode:
                            case 'manual':
                                accept_commit = get_input('Apply this result to project root? (y/n): ').lower() == 'y'
                            case 'auto-accept':
                                accept_commit = True
                            case 'auto-reject':
                                accept_commit = False
                            case _:
                                logger.warning(
                                    f'Unknown commit mode: {commit_mode}; '
                                    f'Fallback to discarding changes.'
                                )
                                accept_commit = False
                        if accept_commit:
                            self.sandbox.commit_winning_branch(winning_branch_path)
                        else:
                            logger.info('Changes discarded.')
                        break
                    else:
                        memory = TypeAdapter(StructuredMemory).validate_python(result['memory'])
                        feedback = result['feedback']
                except KeyboardInterrupt:
                    logger.warning(f'Iteration {index+1} interrupted by user. Stopping...')
                    raise
                except Exception as e:
                    logger.error(f'Iteration {index+1} error: {e}')
            else:
                logger.info('Final result is not deduced, but iteration count is over.')
        except KeyboardInterrupt:
            logger.warning('GCRI Task interrupted by user (Ctrl+C). Returning last state.')
            if result:
                result['final_output'] = 'Task aborted by user.'
            else:
                result = {'final_output': 'Task aborted by user before first iteration completion.'}
        return result
