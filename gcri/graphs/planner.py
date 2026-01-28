import json
import os
from copy import deepcopy as dcp
from datetime import datetime
from typing import Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, START
from loguru import logger
from pydantic import TypeAdapter

from gcri.graphs.gcri_unit import GCRI, TaskAbortedError
from gcri.graphs.schemas import Compression, create_planner_schema
from gcri.graphs.states import StructuredMemory, GlobalState


class GCRIMetaPlanner:
    """
    Meta-level planner that orchestrates multiple GCRI task executions.

    GCRIMetaPlanner breaks down complex, multi-step goals into individual
    tasks, delegates each to a GCRI unit, and compresses learned knowledge
    between steps. It maintains a knowledge context that persists across
    task boundaries.

    Attributes:
        config: Configuration object for planner and GCRI settings.
        schema: Optional Pydantic schema for structured final output.
        abort_event: Optional threading.Event for cooperative cancellation.
        callbacks: Optional GCRICallbacks for environment-specific behavior.
        gcri_unit: Internal GCRI instance for executing individual tasks.
        work_dir: Directory for planner artifacts and logs.
    """

    def __init__(self, config, schema=None, abort_event=None, callbacks=None):
        """
        Initialize the meta-planner with configuration.

        Args:
            config: Configuration with planner, compression, and GCRI settings.
            schema: Optional Pydantic schema for the final planning output.
            abort_event: Optional Event for cooperative task cancellation.
            callbacks: Optional callbacks for commit requests and UI updates.
        """
        self.config = config
        self.schema = schema
        self.abort_event = abort_event
        self.callbacks = callbacks
        gcri_config = dcp(config)
        self.work_dir = os.path.join(
            config.project_dir,
            '.gcri',
            f'planner-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
        gcri_config.run_dir = self.work_dir
        self.gcri_unit = GCRI(gcri_config, abort_event=abort_event, callbacks=callbacks)
        os.makedirs(self.work_dir, exist_ok=True)
        planner_config = config.agents.planner
        planner_schema = create_planner_schema(schema=schema)
        self._planner_agent = init_chat_model(
            planner_config.model_id,
            **planner_config.parameters
        ).with_structured_output(planner_schema)
        compression_config = config.agents.compression
        self._compression_agent = init_chat_model(
            compression_config.model_id,
            **compression_config.parameters
        ).with_structured_output(Compression)
        workflow = StateGraph(GlobalState)
        workflow.add_node('plan', self.plan)
        workflow.add_node('exec_single_gcri_task', self.exec_single_gcri_task)
        workflow.add_node('compress_memory', self.compress_memory)
        workflow.add_edge(START, 'plan')
        workflow.add_conditional_edges('plan', self.router, {'delegate': 'exec_single_gcri_task', 'finish': END})
        workflow.add_edge('exec_single_gcri_task', 'compress_memory')
        workflow.add_edge('compress_memory', 'plan')
        self._workflow = workflow.compile()

    def _check_abort(self):
        """Check if abort has been requested and raise TaskAbortedError if so."""
        if self.abort_event is not None and self.abort_event.is_set():
            logger.warning('🛑 Abort detected in planner. Stopping execution.')
            raise TaskAbortedError('Task aborted by user.')

    @property
    def planner_agent(self):
        return self._planner_agent

    @property
    def compression_agent(self):
        return self._compression_agent

    @property
    def workflow(self):
        return self._workflow

    def _save_state(self, state: GlobalState):
        filename = f'log_plan_{state.plan_count:02d}.json'
        path = os.path.join(self.work_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state.model_dump(mode='json'), f, indent=4, ensure_ascii=False)
        logger.info(f'Result of plan {state.plan_count} saved to: {path}')

    def _broadcast_state(self, state: GlobalState, stage: str):
        logger.bind(
            ui_event='planner_state',
            plan_count=state.plan_count,
            stage=stage
        ).info(f'Planner stage: {stage}')

    def plan(self, state: GlobalState):
        self._check_abort()
        logger.info(f'PLANNING ITER #{state.plan_count} | Analyzing context...')
        self._broadcast_state(state, 'planning')
        
        exec_history = '\n'.join(state.knowledge_context) if state.knowledge_context else 'No prior actions taken.'
        template_path = self.config.templates.planner
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
            schema_desc = 'Provide a comprehensive text summary as the final answer.'
        with open(template_path, 'r') as f:
            template = f.read().format(
                goal=state.goal,
                exec_history=exec_history,
                max_tasks=self.config.plan.num_max_tasks,
                current_step=state.plan_count+1,
                schema_desc=schema_desc
            )
        planning = self.planner_agent.invoke(template)
        logger.info(f'Planner Decision: {planning.is_finished} | Next: {planning.next_task}')
        next_state = {
            'current_task': planning.next_task,
            'final_answer': planning.final_answer
        }
        state_log = state.model_copy(update=next_state)
        self._save_state(state_log)
        self._broadcast_state(state_log, 'planned')
        
        if planning.is_finished:
            return {'final_answer': planning.final_answer, 'current_task': None}
        return {'current_task': planning.next_task, 'final_answer': None}

    def router(self, state: GlobalState) -> Literal['finish', 'delegate']:
        if state.final_answer:
            return 'finish'
        if state.plan_count >= self.config.plan.num_max_tasks:
            logger.warning(f'Max tasks ({self.config.plan.num_max_tasks}) exceeded. Terminating.')
            return 'finish'
        if not state.current_task:
            logger.warning('No task generated and no final answer. Terminating.')
            return 'finish'
        return 'delegate'

    def exec_single_gcri_task(self, state: GlobalState):
        self._check_abort()
        current_task = state.current_task
        logger.info(f'Planning Iter #{state.plan_count} | Delegating to GCRI Unit: {current_task}')
        self._broadcast_state(state, 'executing')
        gcri_result = self.gcri_unit(task=current_task, initial_memory=state.memory, auto_commit=True)
        output = gcri_result.get('final_output', 'Task failed to produce a conclusive final output.')
        updated_memory = gcri_result.get('memory', state.memory)
        return {'mid_result': output, 'memory': updated_memory}

    def compress_memory(self, state: GlobalState):
        self._check_abort()
        logger.info(f'Compress memory of #{state.plan_count}...')
        self._broadcast_state(state, 'compressing')
        raw_constraints = state.memory.active_constraints
        template_path = self.config.templates.compression
        with open(template_path, 'r') as f:
            template = f.read().format(
                task=state.current_task,
                result=state.mid_result,
                active_constraints=json.dumps(raw_constraints, indent=4, ensure_ascii=False)
            )
        compressed = self.compression_agent.invoke(template)
        new_knowledge = f'[Step {state.plan_count+1}] {compressed.summary}'
        current_context = list(state.knowledge_context)
        current_context.append(new_knowledge)
        new_memory = StructuredMemory(
            active_constraints=compressed.retained_constraints,
            history=[]
        )
        logger.info(f'Memory Compressed: {len(raw_constraints)} → {len(compressed.retained_constraints)} constraints.')
        logger.info(f'Knowledge Added: {compressed.summary[:100]}...')

        updated_state_dict = {
            'knowledge_context': current_context,
            'plan_count': state.plan_count+1,
            'current_task': None,
            'mid_result': None,
            'memory': new_memory
        }

        temp_state = state.model_copy(update=updated_state_dict)
        self._save_state(temp_state)
        
        self._broadcast_state(temp_state, 'compressed')

        return updated_state_dict

    def __call__(self, goal_or_state):
        if isinstance(goal_or_state, dict):
            logger.info('🔄 Resuming Planner from in-memory state object...')
            try:
                state = TypeAdapter(GlobalState).validate_python(goal_or_state)
                logger.info(f'Continuing from plan count: {state.plan_count}')
            except Exception as e:
                logger.error(f'Invalid state object: {e}')
                return goal_or_state
        else:
            logger.info(f'Starting meta-planner for goal: {goal_or_state}')
            state = GlobalState(goal=str(goal_or_state))
        self._broadcast_state(state, 'start')
        state = self.workflow.invoke(state)
        self._broadcast_state(state, 'end')
        
        if state['final_answer']:
            logger.info('Goal achieved.')
        else:
            logger.error('Planning logic ended (failed or limit reached).')
        return state
