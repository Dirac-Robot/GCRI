import json
import os
from datetime import datetime
from typing import Literal, List, Optional

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, START
from loguru import logger
from pydantic import BaseModel, Field, TypeAdapter
from copy import deepcopy as dcp

from gcri.graphs.gcri_unit import GCRI
from gcri.graphs.schemas import Plan, Compression
from gcri.graphs.states import StructuredMemory


class GlobalState(BaseModel):
    goal: str
    knowledge_context: List[str] = Field(default_factory=list)
    current_task: Optional[str] = None
    final_answer: Optional[str] = None
    mid_result: Optional[str] = None
    plan_count: int = 0
    memory: StructuredMemory = Field(default_factory=StructuredMemory)


class GCRIMetaPlanner:
    def __init__(self, config):
        self.config = config
        gcri_config = dcp(config)
        self.work_dir = os.path.join(
            config.project_dir,
            '.gcri',
            f'planner-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
        gcri_config.run_dir = self.work_dir
        self.gcri_unit = GCRI(gcri_config)
        os.makedirs(self.work_dir, exist_ok=True)
        planner_config = config.agents.planner
        self._planner_agent = init_chat_model(
            planner_config.model_id,
            **planner_config.parameters
        ).with_structured_output(Plan)
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

    def plan(self, state: GlobalState):
        logger.info(f'PLANNING ITER #{state.plan_count} | Analyzing context...')
        exec_history = '\n'.join(state.knowledge_context) if state.knowledge_context else 'No prior actions taken.'
        template_path = self.config.templates.planner
        with open(template_path, 'r') as f:
            template = f.read().format(
                goal=state.goal,
                exec_history=exec_history,
                max_tasks=self.config.plan.num_max_tasks,
                current_step=state.plan_count+1
            )
        planning = self.planner_agent.invoke(template)
        logger.info(f'Planner Decision: {planning.is_finished} | Next: {planning.next_task}')
        next_state = {
            'current_task': planning.next_task,
            'final_answer': planning.final_answer
        }
        state_log = state.model_copy(update=next_state)
        self._save_state(state_log)
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
        current_task = state.current_task
        logger.info(f'Planning Iter #{state.plan_count} | Delegating to GCRI Unit: {current_task}')
        gcri_result = self.gcri_unit(task=current_task, initial_memory=state.memory, auto_commit=True)
        output = gcri_result.get('final_output', 'Task failed to produce a conclusive final output.')
        updated_memory = gcri_result.get('memory', state.memory)
        return {'mid_result': output, 'memory': updated_memory}

    def compress_memory(self, state: GlobalState):
        logger.info(f'Compress memory of #{state.plan_count}...')
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
        state = self.workflow.invoke(state)
        if state['final_answer']:
            logger.info('Goal achieved.')
        else:
            logger.error('Planning logic ended (failed or limit reached).')
        return state
