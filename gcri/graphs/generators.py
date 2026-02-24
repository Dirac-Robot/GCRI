"""
BranchesGenerator module for GCRI.

Generates RawHypothesis outputs from sampled strategies.
Refinement is handled by the downstream verify-refine micro-loop.
"""
from typing import Protocol, TYPE_CHECKING
from threading import Event

from langgraph.types import Send
from loguru import logger
from pydantic import BaseModel, Field

from gcri.graphs.schemas import Hypothesis, RawHypothesis, Strategies, Strategy, Verification, Refinement
from gcri.graphs.states import TaskState
from gcri.tools.cli import build_model


class TaskAbortedError(Exception):
    """Raised when task is aborted by user."""
    pass


class BranchesGeneratorProtocol(Protocol):

    def generate(self, state: TaskState) -> dict:
        """
        Generate branches with hypotheses.

        Args:
            state: Current TaskState with task and feedback.

        Returns:
            dict with keys:
                - 'strategies': List[Strategy]
                - 'raw_hypotheses': List[RawHypothesis]
                - 'intent_analysis': str
                - 'task_strictness': str
        """
        ...


class BaseBranchesGenerator:

    def __init__(self, config, sandbox, abort_event=None, global_rules=''):
        self.config = config
        self.sandbox = sandbox
        self.abort_event = abort_event
        self.global_rules = global_rules

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

    def _sample_strategies(self, state: TaskState):
        logger.info(f'Iter #{state.count+1} | Request generating strategies...')
        locked_intent = state.intent_analysis or 'None (Analyze Fresh)'
        if state.intent_analysis:
            logger.info(f'Iter #{state.count+1} | Using LOCKED Intent: {locked_intent}')

        logger.bind(
            ui_event='node_update',
            node='strategy',
            data={'type': 'processing'}
        ).info('Generating strategies...')

        strategy_generator_config = self.config.agents.strategy_generator
        strategy_agent = build_model(
            strategy_generator_config.model_id,
            strategy_generator_config.get('gcri_options'),
            **strategy_generator_config.parameters
        ).with_structured_output(schema=Strategies)

        template = self._load_template_with_rules(
            self.config.templates.strategy_generator,
            task=state.task,
            feedback=state.feedback,
            num_hypothesis=len(self.config.agents.branches),
            locked_intent=locked_intent
        )
        strategies = self._invoke_with_retry(strategy_agent, template, 'Strategy agent')

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


class DefaultBranchesGenerator(BaseBranchesGenerator):

    def _generate_branch(self, index, task, strategy, intent_analysis, strictness, feedback, count, base_container_id=None):
        self._check_abort()

        if base_container_id:
            container_id = self.sandbox.setup_branch_from_base(count, index, base_container_id)
        else:
            container_id = self.sandbox.setup_branch(count, index)

        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=index,
            data={'type': 'processing'}
        ).info(f'Iter #{count+1} | Branch[{index}] Sampling hypothesis...')

        hypothesis_config = self.config.agents.branches[index].hypothesis
        hypothesis_agent = build_model(
            hypothesis_config.model_id,
            hypothesis_config.get('gcri_options'),
            container_id=container_id,
            **hypothesis_config.parameters
        ).with_structured_output(schema=Hypothesis)

        hypothesis_template = self._load_template_with_rules(
            self.config.templates.hypothesis,
            task=task,
            strictness=strictness,
            strategy=strategy,
            intent_analysis=intent_analysis
        )
        hypothesis = self._invoke_with_retry(
            hypothesis_agent, hypothesis_template, f'Hypothesis agent at strategy #{index+1}'
        )

        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=index,
            data={'hypothesis': hypothesis.hypothesis, 'container_id': container_id}
        ).info(f'Iter #{count+1} | Branch[{index}] Hypothesis: {hypothesis.hypothesis[:80]}...')

        current_hypothesis = hypothesis.hypothesis
        adjustment_log = ''

        max_verify_iters = self.config.protocols.get('max_verifying_iterations', 3)
        for verify_round in range(max_verify_iters):
            self._check_abort()

            logger.bind(
                ui_event='node_update', node='verification',
                branch=index, data={'type': 'processing'}
            ).info(f'Iter #{count+1} | Branch[{index}] Verifying (round {verify_round+1})...')

            verification_config = self.config.agents.verification
            if index < len(self.config.agents.branches):
                branch_cfg = self.config.agents.branches[index]
                if hasattr(branch_cfg, 'verification') and branch_cfg.verification.get('model_id'):
                    verification_config = branch_cfg.verification

            verify_agent = build_model(
                verification_config.model_id,
                verification_config.get('gcri_options'),
                container_id=container_id,
                **verification_config.parameters
            ).with_structured_output(schema=Verification)

            verify_template = self._load_template_with_rules(
                self.config.templates.verification,
                task=task,
                strategy=strategy,
                reasoning='',
                hypothesis=current_hypothesis,
                intent_analysis=intent_analysis
            )
            verification = self._invoke_with_retry(verify_agent, verify_template, 'Verification agent')

            logger.bind(
                ui_event='node_update', node='verification',
                branch=index, data={
                    'counter_example': verification.counter_example,
                    'counter_strength': verification.counter_strength,
                    'container_id': container_id
                }
            ).info(
                f'Iter #{count+1} | Branch[{index}] Verification: '
                f'{verification.counter_strength.upper()} counter'
            )

            if verification.counter_strength not in ('strong', 'weak'):
                break

            self._check_abort()

            logger.bind(
                ui_event='node_update', node='refinement',
                branch=index, data={'type': 'processing'}
            ).info(f'Iter #{count+1} | Branch[{index}] Refining against {verification.counter_strength} counter-example...')

            refinement_config = self.config.agents.refinement
            if index < len(self.config.agents.branches):
                branch_cfg = self.config.agents.branches[index]
                if hasattr(branch_cfg, 'refinement') and branch_cfg.refinement.get('model_id'):
                    refinement_config = branch_cfg.refinement

            refine_agent = build_model(
                refinement_config.model_id,
                refinement_config.get('gcri_options'),
                container_id=container_id,
                **refinement_config.parameters
            ).with_structured_output(schema=Refinement)

            refine_template = self._load_template_with_rules(
                self.config.templates.refinement,
                task=task,
                strategy=strategy,
                intent_analysis=intent_analysis,
                hypothesis=current_hypothesis,
                counter_example=verification.counter_example,
                counter_reasoning=verification.reasoning
            )
            refinement = self._invoke_with_retry(refine_agent, refine_template, 'Refinement agent')

            current_hypothesis = refinement.refined_hypothesis
            adjustment_log += f'[Round {verify_round+1}]: {refinement.adjustment_log}\n'

            logger.bind(
                ui_event='node_update', node='refinement',
                branch=index, data={'adjustment': refinement.adjustment_log, 'container_id': container_id}
            ).info(f'Iter #{count+1} | Branch[{index}] Refined: {refinement.adjustment_log}')

        return RawHypothesis(
            index=index,
            strategy_name=strategy.name,
            strategy_description=strategy.description,
            hypothesis=current_hypothesis,
            reasoning=adjustment_log.strip(),
            container_id=container_id
        )

    def generate(self, state: TaskState) -> dict:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        strategies_result = self._sample_strategies(state)

        logger.bind(ui_event='phase_change', phase='hypothesis_generation').info(
            'Starting Hypothesis Generation...'
        )
        num_branches = min(len(self.config.agents.branches), len(strategies_result['strategies']))
        logger.info(f'🌿 Spawning {num_branches} hypothesis branches...')

        base_container = getattr(state.memory, 'base_sandbox_container_id', None)
        if base_container:
            logger.info(f'📦 Using base sandbox {base_container[:12]} for all branches')

        raw_hypotheses = [None]*num_branches

        with ThreadPoolExecutor(max_workers=num_branches) as executor:
            futures = {
                executor.submit(
                    self._generate_branch,
                    index=i,
                    task=state.task,
                    strategy=strategies_result['strategies'][i],
                    intent_analysis=strategies_result['intent_analysis'],
                    strictness=strategies_result['task_strictness'],
                    feedback=state.feedback or '',
                    count=state.count,
                    base_container_id=base_container
                ): i for i in range(num_branches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                raw_hypotheses[idx] = future.result()

        return {
            'strategies': strategies_result['strategies'],
            'intent_analysis': strategies_result['intent_analysis'],
            'task_strictness': strategies_result['task_strictness'],
            'raw_hypotheses': raw_hypotheses
        }
