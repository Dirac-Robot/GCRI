import json
import os
import shutil
from datetime import datetime

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


class GCRI:
    def __init__(self, config, schema=None):
        self.config = config
        self.schema = schema
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
        self._project_dir = config.project_dir
        self._run_dir = config.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self._work_dir = None
        self._log_dir = None
        if schema:
            decision_schema = create_decision_schema(schema=schema)
            logger.info(f"🔧 Custom output schema applied: {decision_schema.__name__}")
        else:
            decision_schema = DecisionProtoType
        decision_agent = build_model(
            decision_config.model_id,
            decision_config.get('gcri_options'),
            work_dir=self._work_dir,
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

    def _setup_sandbox(self):
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._work_dir = os.path.join(self.run_dir, f'run-{timestamp}')
        self._log_dir = os.path.join(self.work_dir, f'logs')
        logger.info(f"📦 Creating workspaces in sandbox at: {self.work_dir}")
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def project_dir(self):
        return self._project_dir

    @property
    def run_dir(self):
        return self._run_dir

    @property
    def work_dir(self):
        return self._work_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def graph(self):
        return self._graph

    @property
    def workflow(self):
        return self._workflow

    def _smart_copy(self, src, dst, *, follow_symlinks=True):
        limit_bytes = self.config.protocols.max_copy_size*1024*1024  # 10MB
        try:
            if os.path.islink(src):
                link_to = os.readlink(src)
                os.symlink(link_to, dst)
            elif os.path.getsize(src) > limit_bytes:
                os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copy2(src, dst)
        except Exception as e:
            logger.warning(f'Smart copy failed for {src}: {e}')

    def map_branches(self, state: TaskState):
        num_branches = min(len(self.config.agents.branches), len(state.strategies))
        root_dir = os.path.join(self.work_dir, 'workspaces')
        os.makedirs(root_dir, exist_ok=True)
        sends = []
        ignore = shutil.ignore_patterns(
            '.git',
            '__pycache__',
            'venv',
            'env',
            'node_modules',
            '.idea',
            '.vscode',
            '.gcri',
            '*.pyc'
        )
        for index in range(num_branches):
            branch_workspace = os.path.join(root_dir, f'iter_{state.count}_branch_{index}')
            if os.path.exists(branch_workspace):
                shutil.rmtree(branch_workspace)
            os.makedirs(branch_workspace, exist_ok=True)
            shutil.copytree(
                self.project_dir,
                branch_workspace,
                ignore=ignore,
                copy_function=self._smart_copy,
                dirs_exist_ok=True
            )
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
            logger.info(f'Iter #{state.count+1} | Sampled strategy #{index}: {strategy}')
        return dict(task_strictness=strategies.strictness, strategies=strategies.strategies)

    def sample_hypothesis(self, state: BranchState):
        logger.info(f'Iter #{state.count_in_branch+1} | Request sampling hypothesis for strategy #{state.index}...')
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
                f'at strategy #{state.index}.'
            )
        logger.info(f'Iter #{state.count_in_branch+1} | Sampled hypothesis #{state.index}: {hypothesis.hypothesis}')
        return dict(hypothesis=hypothesis.hypothesis)

    def reasoning_and_refine(self, state: BranchState):
        logger.info(f'Iter #{state.count_in_branch+1} | Request reasoning and refining hypothesis #{state.index}...')
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
                f'at hypothesis #{state.index}.'
            )
        logger.info(
            f'Iter #{state.count_in_branch+1} | '
            f'Refined hypothesis #{state.index}: {reasoning.refined_hypothesis}'
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
        logger.info(f'Iter #{state.count_in_branch+1} | Request verifying refined hypothesis #{state.index}...')
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
            f'Counter-Example of Hypothesis #{state.index}(Counter Strength: {verification.counter_strength}): '
            f'{verification.counter_example}'
        )
        return {'results': [result]}

    @classmethod
    def _get_failure_category_description(cls):
        descriptions = []
        for code in FailureCategory:
            descriptions.append(f'- {code.value}')
        return '\n'.join(descriptions)

    @classmethod
    def _commit_winning_branch(cls, winning_branch_path, project_root):
        logger.info(f'💾 Merging changes from winning branch: {winning_branch_path}')
        ignore_patterns = {'.git', '__pycache__', 'venv', 'env', '.idea', 'workspaces'}
        for root, dirs, files in os.walk(winning_branch_path):
            dirs[:] = [d for d in dirs if d not in ignore_patterns]
            rel_path = os.path.relpath(root, winning_branch_path)
            target_dir = os.path.join(project_root, rel_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            for file in files:
                if file in ignore_patterns or file.endswith('.pyc'):
                    continue
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, file)
                if os.path.islink(src_file):
                    continue
                try:
                    shutil.copy2(src_file, dst_file)
                except Exception as e:
                    logger.error(f'Failed to copy {src_file}: {e}')
        logger.info('✅ Merge completed successfully.')

    def decide(self, state: TaskState):
        logger.info(f'Iter #{state.count+1} | Request generating final decision for current loop...')
        file_contexts = []
        num_results = len(state.results)
        workspace_root = os.path.join(self.work_dir, 'workspaces')
        for i in range(num_results):
            branch_dir = os.path.join(workspace_root, f'iter_{state.count}_branch_{i}')
            if os.path.exists(branch_dir):
                rel_path = os.path.relpath(branch_dir, start=self.work_dir)
                file_contexts.append(f'- Branch {i} files location: {rel_path}')
            else:
                file_contexts.append(f'- Branch {i} files location: (Directory not found)')
        file_contexts = '\n'.join(file_contexts)
        template_path = self.config.templates.decision
        with open(template_path, 'r') as f:
            template = f.read()
        if state.aggregated_result:
            aggregated_result = json.dumps(state.aggregated_result, indent=4, ensure_ascii=False)
        else:
            aggregated_result = None
        if self.schema:
            schema_desc = (
                'MUST follow the specific JSON schema provided in the tool definition. '
                'Populate fields based on the winning branch.'
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
            logger.info(f'Selected Best Branch Index: {decision.best_branch_index}')
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

    def __call__(self, task, initial_memory=None, auto_commit=False):
        self._setup_sandbox()
        self.decision_agent.work_dir = self.work_dir
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
        try:
            for index in range(self.config.max_iterations):
                logger.info(f'Starting Iteration {index}...')
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
                    os.makedirs(self.log_dir, exist_ok=True)
                    log_path = os.path.join(self.log_dir, f'log_iteration_{index:02d}.json')
                    with open(log_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=4, ensure_ascii=False)
                    logger.info(f'Result of iteration {index} saved to: {log_path}')
                    if result['decision']:
                        logger.info('Final result is successfully deduced.')
                        logger.info(f'Task Completed. Check sandbox: {self.work_dir}')
                        best_branch_index = result.get('best_branch_index')
                        if best_branch_index is None:
                            logger.warning(
                                'Decision is True but no branch index provided. Cannot commit automatically.'
                            )
                            break
                        winning_branch_path = os.path.join(
                            self.work_dir, 'workspaces', f'iter_{index}_branch_{best_branch_index}'
                        )
                        logger.info(f'🏆 Winning Branch Identified: Branch #{best_branch_index}')
                        logger.info(f'📂 Location: {winning_branch_path}')
                        if auto_commit or get_input('Apply this result to project root? (y/n): ').lower() == 'y':
                            self._commit_winning_branch(winning_branch_path, self.project_dir)
                        else:
                            logger.info('Changes discarded.')
                        break
                    else:
                        memory = TypeAdapter(StructuredMemory).validate_python(result['memory'])
                        feedback = result['feedback']
                except KeyboardInterrupt:
                    logger.warning(f'Iteration {index} interrupted by user. Stopping...')
                    raise
                except Exception as e:
                    logger.error(f'Iteration {index} error: {e}')
            else:
                logger.info('Final result is not deduced, but iteration count is over.')
        except KeyboardInterrupt:
            logger.warning('GCRI Task interrupted by user (Ctrl+C). Returning last state.')
            if result:
                result['final_output'] = 'Task aborted by user.'
            else:
                result = {'final_output': 'Task aborted by user before first iteration completion.'}
        return result
