"""
BranchesGenerator module for GCRI.

Provides interchangeable branch generation strategies that produce
RawHypothesis outputs from strategies. The default implementation
mirrors the original hypothesis + reasoning_and_refine workflow.

Generators are independent from GCRI - they receive config, sandbox,
abort_event, and global_rules directly.
"""
from typing import Protocol, TYPE_CHECKING
from threading import Event

from langgraph.types import Send
from loguru import logger
from pydantic import BaseModel, Field

from gcri.graphs.schemas import Hypothesis, Reasoning, RawHypothesis, Strategies, Strategy
from gcri.graphs.states import TaskState
from gcri.tools.cli import build_model


class TaskAbortedError(Exception):
    """Raised when task is aborted by user."""
    pass


class BranchesGeneratorProtocol(Protocol):
    """Protocol for interchangeable branch generators."""

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
    """Base class for all branch generators with common helpers."""

    def __init__(self, config, sandbox, abort_event=None, global_rules=''):
        self.config = config
        self.sandbox = sandbox
        self.abort_event = abort_event
        self.global_rules = global_rules

    def _check_abort(self):
        """Check if abort has been requested."""
        if self.abort_event is not None and self.abort_event.is_set():
            logger.warning('🛑 Abort detected. Stopping execution.')
            raise TaskAbortedError('Task aborted by user.')

    def _load_template_with_rules(self, template_path, **format_kwargs):
        """Load template file, format with kwargs, and prepend global rules."""
        with open(template_path, 'r') as f:
            template = f.read().format(**format_kwargs)
        return f'{self.global_rules}\n\n{template}'

    def _invoke_with_retry(self, agent, template, error_context='agent'):
        """Invoke agent with retry logic."""
        for _ in range(self.config.protocols.max_tries_per_agent):
            result = agent.invoke(template)
            if result is not None:
                return result
        raise ValueError(
            f'{error_context} could not generate output '
            f'for {self.config.protocols.max_tries_per_agent} times.'
        )

    def _sample_strategies(self, state: TaskState):
        """Sample strategies for the task."""
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
        """Generate hypothesis with reasoning refinement for a single branch."""
        self._check_abort()

        if base_container_id:
            container_id = self.sandbox.setup_branch_from_base(count, index, base_container_id)
        else:
            container_id = self.sandbox.setup_branch(count, index)

        # Step 1: Hypothesis
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

        # Step 2: Reasoning & Refine
        self._check_abort()
        logger.bind(
            ui_event='node_update',
            node='reasoning',
            branch=index,
            data={'type': 'processing'}
        ).info(f'Iter #{count+1} | Branch[{index}] Reasoning...')

        reasoning_config = self.config.agents.branches[index].reasoning
        reasoning_agent = build_model(
            reasoning_config.model_id,
            reasoning_config.get('gcri_options'),
            container_id=container_id,
            **reasoning_config.parameters
        ).with_structured_output(schema=Reasoning)

        reasoning_template = self._load_template_with_rules(
            self.config.templates.reasoning,
            task=task,
            strategy=strategy,
            hypothesis=hypothesis.hypothesis,
            intent_analysis=intent_analysis
        )
        reasoning = self._invoke_with_retry(
            reasoning_agent, reasoning_template, f'Reasoning agent at hypothesis #{index+1}'
        )

        raw_hyp = RawHypothesis(
            index=index,
            strategy_name=strategy.name,
            strategy_description=strategy.description,
            hypothesis=reasoning.refined_hypothesis,
            reasoning=reasoning.reasoning,
            container_id=container_id
        )

        logger.bind(
            ui_event='node_update',
            node='reasoning',
            branch=index,
            data={
                'reasoning': reasoning.reasoning,
                'hypothesis': reasoning.refined_hypothesis,
                'container_id': container_id
            }
        ).info(f'Iter #{count+1} | Branch[{index}] Refined: {reasoning.refined_hypothesis[:80]}...')

        return raw_hyp

    def generate(self, state: TaskState) -> dict:
        """Generate strategies and hypotheses."""
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


class DeepThinkGenerator(BaseBranchesGenerator):
    """
    Deep thinking branch generator with full refinement workflow.

    Executes: sample_strategies → parallel(hypothesis + reasoning_and_refine)

    This is equivalent to the DefaultBranchesGenerator, providing the
    standard GCRI workflow with strategy sampling, hypothesis generation,
    and reasoning-based refinement.
    """

    def _generate_branch(self, index, task, strategy, intent_analysis, strictness, feedback, count, base_container_id=None):
        """Generate hypothesis with reasoning refinement for a single branch."""
        self._check_abort()

        if base_container_id:
            container_id = self.sandbox.setup_branch_from_base(count, index, base_container_id)
        else:
            container_id = self.sandbox.setup_branch(count, index)

        # Step 1: Hypothesis
        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=index,
            data={'type': 'processing'}
        ).info(f'[DeepThink] Iter #{count+1} | Branch[{index}] Sampling hypothesis...')

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
        ).info(f'[DeepThink] Iter #{count+1} | Branch[{index}] Hypothesis: {hypothesis.hypothesis[:80]}...')

        # Step 2: Reasoning & Refine
        self._check_abort()
        logger.bind(
            ui_event='node_update',
            node='reasoning',
            branch=index,
            data={'type': 'processing'}
        ).info(f'[DeepThink] Iter #{count+1} | Branch[{index}] Reasoning and refining...')

        reasoning_config = self.config.agents.branches[index].reasoning
        reasoning_agent = build_model(
            reasoning_config.model_id,
            reasoning_config.get('gcri_options'),
            container_id=container_id,
            **reasoning_config.parameters
        ).with_structured_output(schema=Reasoning)

        reasoning_template = self._load_template_with_rules(
            self.config.templates.reasoning,
            task=task,
            strategy=strategy,
            hypothesis=hypothesis.hypothesis,
            intent_analysis=intent_analysis
        )
        reasoning = self._invoke_with_retry(
            reasoning_agent, reasoning_template, f'Reasoning agent at hypothesis #{index+1}'
        )

        raw_hyp = RawHypothesis(
            index=index,
            strategy_name=strategy.name,
            strategy_description=strategy.description,
            hypothesis=reasoning.refined_hypothesis,
            reasoning=reasoning.reasoning,
            container_id=container_id
        )

        logger.bind(
            ui_event='node_update',
            node='reasoning',
            branch=index,
            data={
                'reasoning': reasoning.reasoning,
                'hypothesis': reasoning.refined_hypothesis,
                'container_id': container_id
            }
        ).info(f'[DeepThink] Iter #{count+1} | Branch[{index}] Refined: {reasoning.refined_hypothesis[:80]}...')

        return raw_hyp

    def generate(self, state: TaskState) -> dict:
        """Generate strategies and hypotheses with full refinement."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        strategies_result = self._sample_strategies(state)

        logger.bind(ui_event='phase_change', phase='hypothesis_generation').info(
            '[DeepThink] Starting Hypothesis Generation...'
        )
        num_branches = min(len(self.config.agents.branches), len(strategies_result['strategies']))
        logger.info(f'🌿 [DeepThink] Spawning {num_branches} hypothesis branches...')

        base_container = getattr(state.memory, 'base_sandbox_container_id', None)
        if base_container:
            logger.info(f'📦 [DeepThink] Using base sandbox {base_container[:12]} for all branches')

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


class LowThinkGenerator(BaseBranchesGenerator):
    """
    Low thinking branch generator for fast hypothesis generation.

    Executes: sample_strategies → parallel(hypothesis only)

    Skips the reasoning/refine step for faster execution. The raw hypothesis
    is used directly without refinement.
    """

    # Uses parent's __init__, no workflow needed (ThreadPoolExecutor pattern)

    def _generate_branch_hypothesis(self, index, task, strategy, intent_analysis, strictness, feedback, count, base_container_id=None):
        """Generate hypothesis for a single branch - callable directly from ThreadPoolExecutor."""
        self._check_abort()

        # Setup container - use base sandbox if available
        if base_container_id:
            container_id = self.sandbox.setup_branch_from_base(count, index, base_container_id)
        else:
            container_id = self.sandbox.setup_branch(count, index)

        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=index,
            data={'type': 'processing'}
        ).info(f'[LowThink] Iter #{count+1} | Branch[{index}] Sampling hypothesis (no refine)...')

        hypothesis_config = self.config.agents.branches[index].hypothesis
        agent = build_model(
            hypothesis_config.model_id,
            hypothesis_config.get('gcri_options'),
            container_id=container_id,
            **hypothesis_config.parameters
        ).with_structured_output(schema=Hypothesis)

        template = self._load_template_with_rules(
            self.config.templates.hypothesis,
            task=task,
            strictness=strictness,
            strategy=strategy,
            intent_analysis=intent_analysis
        )
        hypothesis = self._invoke_with_retry(
            agent, template, f'Hypothesis agent at strategy #{index+1}'
        )

        # Directly create RawHypothesis without reasoning step
        raw_hyp = RawHypothesis(
            index=index,
            strategy_name=strategy.name,
            strategy_description=strategy.description,
            hypothesis=hypothesis.hypothesis,
            reasoning='[LowThink: No refinement applied]',
            container_id=container_id
        )

        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=index,
            data={'hypothesis': hypothesis.hypothesis, 'container_id': container_id}
        ).info(f'[LowThink] Iter #{count+1} | Branch[{index}] Hypothesis: {hypothesis.hypothesis[:80]}...')

        return raw_hyp

    def _map_branches(self, state: TaskState):
        """Map strategies to parallel branch executions."""
        logger.bind(ui_event='phase_change', phase='hypothesis_generation').info(
            '[LowThink] Starting Fast Hypothesis Generation...'
        )
        num_branches = min(len(self.config.agents.branches), len(state.strategies))
        logger.info(f'🌿 [LowThink] Spawning {num_branches} hypothesis branches (no refine)...')

        sends = []
        for index in range(num_branches):
            container_id = self.sandbox.setup_branch(state.count, index)
            sends.append(Send('branch_executor', {
                'index': index,
                'count_in_branch': state.count,
                'task_in_branch': state.task,
                'strictness': state.task_strictness,
                'strategy': state.strategies[index],
                'feedback': state.feedback,
                'container_id': container_id,
                'intent_analysis_in_branch': state.intent_analysis
            }))
        return sends

    def generate(self, state: TaskState) -> dict:
        """Generate strategies and hypotheses without refinement."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # First, sample strategies
        strategies_result = self._sample_strategies(state)

        logger.bind(ui_event='phase_change', phase='hypothesis_generation').info(
            '[LowThink] Starting Fast Hypothesis Generation...'
        )
        num_branches = min(len(self.config.agents.branches), len(strategies_result['strategies']))
        logger.info(f'🌿 [LowThink] Spawning {num_branches} hypothesis branches (no refine)...')

        # Check for base sandbox from previous iteration
        base_container = getattr(state.memory, 'base_sandbox_container_id', None)
        if base_container:
            logger.info(f'📦 [LowThink] Using base sandbox {base_container[:12]} for all branches')

        raw_hypotheses = [None]*num_branches

        with ThreadPoolExecutor(max_workers=num_branches) as executor:
            futures = {
                executor.submit(
                    self._generate_branch_hypothesis,
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
                raw_hyp = future.result()
                raw_hypotheses[idx] = raw_hyp

        return {
            'strategies': strategies_result['strategies'],
            'intent_analysis': strategies_result['intent_analysis'],
            'task_strictness': strategies_result['task_strictness'],
            'raw_hypotheses': raw_hypotheses
        }


class AdaptiveThinkGenerator(BaseBranchesGenerator):
    """
    Adaptive branch generator that decides refinement based on task strictness.

    - strictness='strict' → Full refinement (DeepThink behavior)
    - strictness='moderate' or 'creative' → No refinement (LowThink behavior)

    This eliminates the need to manually choose between DeepThink and LowThink,
    as the decision is made automatically based on task analysis.
    """

    def _generate_branch_with_refine(self, index, task, strategy, intent_analysis, strictness, feedback, count, base_container_id=None):
        """Generate hypothesis with reasoning refinement (DeepThink path)."""
        self._check_abort()

        # Setup container
        if base_container_id:
            container_id = self.sandbox.setup_branch_from_base(count, index, base_container_id)
        else:
            container_id = self.sandbox.setup_branch(count, index)

        # Step 1: Generate hypothesis
        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=index,
            data={'type': 'processing'}
        ).info(f'[Adaptive:Strict] Iter #{count+1} | Branch[{index}] Sampling hypothesis...')

        hypothesis_config = self.config.agents.branches[index].hypothesis
        hyp_agent = build_model(
            hypothesis_config.model_id,
            hypothesis_config.get('gcri_options'),
            container_id=container_id,
            **hypothesis_config.parameters
        ).with_structured_output(schema=Hypothesis)

        hyp_template = self._load_template_with_rules(
            self.config.templates.hypothesis,
            task=task,
            strictness=strictness,
            strategy=strategy,
            intent_analysis=intent_analysis
        )
        hypothesis = self._invoke_with_retry(
            hyp_agent, hyp_template, f'Hypothesis agent at strategy #{index+1}'
        )

        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=index,
            data={'hypothesis': hypothesis.hypothesis, 'container_id': container_id}
        ).info(f'[Adaptive:Strict] Iter #{count+1} | Branch[{index}] Hypothesis: {hypothesis.hypothesis[:80]}...')

        # Step 2: Reasoning and refine
        self._check_abort()
        logger.bind(
            ui_event='node_update',
            node='reasoning',
            branch=index,
            data={'type': 'processing'}
        ).info(f'[Adaptive:Strict] Iter #{count+1} | Branch[{index}] Reasoning and refining...')

        reasoning_config = self.config.agents.branches[index].reasoning
        reason_agent = build_model(
            reasoning_config.model_id,
            reasoning_config.get('gcri_options'),
            container_id=container_id,
            **reasoning_config.parameters
        ).with_structured_output(schema=Reasoning)

        reason_template = self._load_template_with_rules(
            self.config.templates.reasoning,
            task=task,
            strategy=strategy,
            hypothesis=hypothesis.hypothesis,
            intent_analysis=intent_analysis
        )
        reasoning = self._invoke_with_retry(
            reason_agent, reason_template, f'Reasoning agent at hypothesis #{index+1}'
        )

        raw_hyp = RawHypothesis(
            index=index,
            strategy_name=strategy.name,
            strategy_description=strategy.description,
            hypothesis=reasoning.refined_hypothesis,
            reasoning=reasoning.reasoning,
            container_id=container_id
        )

        logger.bind(
            ui_event='node_update',
            node='reasoning',
            branch=index,
            data={
                'reasoning': reasoning.reasoning,
                'hypothesis': reasoning.refined_hypothesis,
                'container_id': container_id
            }
        ).info(f'[Adaptive:Strict] Iter #{count+1} | Branch[{index}] Refined: {reasoning.refined_hypothesis[:80]}...')

        return raw_hyp

    def _generate_branch_no_refine(self, index, task, strategy, intent_analysis, strictness, feedback, count, base_container_id=None):
        """Generate hypothesis without refinement (LowThink path)."""
        self._check_abort()

        # Setup container
        if base_container_id:
            container_id = self.sandbox.setup_branch_from_base(count, index, base_container_id)
        else:
            container_id = self.sandbox.setup_branch(count, index)

        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=index,
            data={'type': 'processing'}
        ).info(f'[Adaptive:Fast] Iter #{count+1} | Branch[{index}] Sampling hypothesis (no refine)...')

        hypothesis_config = self.config.agents.branches[index].hypothesis
        agent = build_model(
            hypothesis_config.model_id,
            hypothesis_config.get('gcri_options'),
            container_id=container_id,
            **hypothesis_config.parameters
        ).with_structured_output(schema=Hypothesis)

        template = self._load_template_with_rules(
            self.config.templates.hypothesis,
            task=task,
            strictness=strictness,
            strategy=strategy,
            intent_analysis=intent_analysis
        )
        hypothesis = self._invoke_with_retry(
            agent, template, f'Hypothesis agent at strategy #{index+1}'
        )

        raw_hyp = RawHypothesis(
            index=index,
            strategy_name=strategy.name,
            strategy_description=strategy.description,
            hypothesis=hypothesis.hypothesis,
            reasoning='[Adaptive:Fast - No refinement applied]',
            container_id=container_id
        )

        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=index,
            data={'hypothesis': hypothesis.hypothesis, 'container_id': container_id}
        ).info(f'[Adaptive:Fast] Iter #{count+1} | Branch[{index}] Hypothesis: {hypothesis.hypothesis[:80]}...')

        return raw_hyp

    def generate(self, state: TaskState) -> dict:
        """Generate strategies and hypotheses with adaptive refinement."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Step 1: Sample strategies (this also determines strictness)
        strategies_result = self._sample_strategies(state)
        strictness = strategies_result['task_strictness']

        # Step 2: Decide refinement based on strictness
        should_refine = (strictness == 'strict')
        mode_label = 'Strict (with refine)' if should_refine else 'Fast (no refine)'

        logger.bind(ui_event='phase_change', phase='hypothesis_generation').info(
            f'[Adaptive] Starting Generation - Mode: {mode_label}'
        )
        logger.info(f'📊 [Adaptive] Strictness={strictness} → Refine={should_refine}')

        num_branches = min(len(self.config.agents.branches), len(strategies_result['strategies']))
        logger.info(f'🌿 [Adaptive] Spawning {num_branches} hypothesis branches...')

        # Check for base sandbox
        base_container = getattr(state.memory, 'base_sandbox_container_id', None)
        if base_container:
            logger.info(f'📦 [Adaptive] Using base sandbox {base_container[:12]} for all branches')

        raw_hypotheses = [None]*num_branches

        # Choose the appropriate generation method based on strictness
        generate_fn = self._generate_branch_with_refine if should_refine else self._generate_branch_no_refine

        with ThreadPoolExecutor(max_workers=num_branches) as executor:
            futures = {
                executor.submit(
                    generate_fn,
                    index=i,
                    task=state.task,
                    strategy=strategies_result['strategies'][i],
                    intent_analysis=strategies_result['intent_analysis'],
                    strictness=strictness,
                    feedback=state.feedback or '',
                    count=state.count,
                    base_container_id=base_container
                ): i for i in range(num_branches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                raw_hyp = future.result()
                raw_hypotheses[idx] = raw_hyp

        return {
            'strategies': strategies_result['strategies'],
            'intent_analysis': strategies_result['intent_analysis'],
            'task_strictness': strictness,
            'raw_hypotheses': raw_hypotheses
        }


class MinimalBranchOutput(BaseModel):
    """Combined output for MinimalThinkGenerator - strategy and hypothesis in one shot."""
    strategy_name: str = Field(..., description='A short name for the approach taken')
    strategy_description: str = Field(..., description='Brief description of the strategy used')
    hypothesis: str = Field(..., description='The complete solution/answer')


class MinimalThinkGenerator(BaseBranchesGenerator):
    """
    Minimal thinking branch generator - fastest path to hypotheses.

    Each branch directly generates strategy + hypothesis in a single LLM call.
    No separate strategy sampling, no reasoning/refine steps.
    """

    def __init__(self, config, sandbox, abort_event=None, global_rules=''):
        super().__init__(config, sandbox, abort_event, global_rules)

    def _generate_branch(self, index: int, task: str, feedback: str, count: int, base_container_id: str = None) -> RawHypothesis:
        """Generate strategy + hypothesis for a single branch."""
        self._check_abort()

        # Setup container - use base sandbox if available
        if base_container_id:
            container_id = self.sandbox.setup_branch_from_base(count, index, base_container_id)
        else:
            container_id = self.sandbox.setup_branch(count, index)

        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=index,
            data={'type': 'processing'}
        ).info(f'[MinimalThink] Iter #{count+1} | Branch[{index}] Generating...')

        hypothesis_config = self.config.agents.branches[index].hypothesis
        agent = build_model(
            hypothesis_config.model_id,
            hypothesis_config.get('gcri_options'),
            container_id=container_id,
            **hypothesis_config.parameters
        ).with_structured_output(schema=MinimalBranchOutput)

        template = self._load_template_with_rules(
            self.config.templates.hypothesis_minimal,
            task=task,
            intent_analysis=feedback or 'Solve the given task directly.'
        )
        result = self._invoke_with_retry(
            agent, template, f'MinimalThink branch #{index+1}'
        )

        raw_hyp = RawHypothesis(
            index=index,
            strategy_name=result.strategy_name,
            strategy_description=result.strategy_description,
            hypothesis=result.hypothesis,
            reasoning='[MinimalThink: Single-shot generation]',
            container_id=container_id
        )

        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=index,
            data={'hypothesis': result.hypothesis, 'container_id': container_id}
        ).info(f'[MinimalThink] Iter #{count+1} | Branch[{index}] Done: {result.hypothesis[:80]}...')

        return raw_hyp

    def generate(self, state: TaskState) -> dict:
        """Generate hypotheses - single branch only (no strategy agent for diversity)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.bind(ui_event='phase_change', phase='hypothesis_generation').info(
            '[MinimalThink] Starting Minimal Generation...'
        )

        # MinimalThink has no Strategy Agent, so multiple branches converge to similar results
        configured_branches = len(self.config.agents.branches)
        if configured_branches > 1:
            logger.warning(
                f'⚠️ [MinimalThink] {configured_branches} branches configured, but MinimalThink has no Strategy Agent. '
                f'Multiple branches will produce similar results. Forcing to 1 branch. '
                f'Use LowThinkGenerator or DeepThinkGenerator for branch diversity.'
            )
        num_branches = 1  # Always 1 branch for MinimalThink
        logger.info(f'🌿 [MinimalThink] Spawning {num_branches} branch...')

        # Check for base sandbox from previous iteration
        base_container = getattr(state.memory, 'base_sandbox_container_id', None)
        if base_container:
            logger.info(f'📦 [MinimalThink] Using base sandbox {base_container[:12]}')

        raw_hypotheses = [None]*num_branches
        strategies = [None]*num_branches

        with ThreadPoolExecutor(max_workers=num_branches) as executor:
            futures = {
                executor.submit(
                    self._generate_branch,
                    index=i,
                    task=state.task,
                    feedback=state.feedback or '',
                    count=state.count,
                    base_container_id=base_container
                ): i for i in range(num_branches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                raw_hyp = future.result()
                raw_hypotheses[idx] = raw_hyp
                strategies[idx] = Strategy(
                    name=raw_hyp.strategy_name,
                    description=raw_hyp.strategy_description,
                    feedback_reflection=state.feedback or '',
                    hints=[]
                )

        return {
            'strategies': strategies,
            'intent_analysis': '',
            'task_strictness': 'moderate',
            'raw_hypotheses': raw_hypotheses
        }


def get_branches_generator(generator_type: str, config, sandbox, abort_event=None, global_rules=''):
    """Factory function to get the appropriate BranchesGenerator.

    Args:
        generator_type: One of 'default', 'deep', 'low', 'adaptive', 'minimal'
        config: Configuration object
        sandbox: SandboxManager instance
        abort_event: Optional threading.Event for cooperative cancellation
        global_rules: Global rules string for prompts

    Returns:
        BranchesGenerator instance
    """
    generators = {
        'default': DefaultBranchesGenerator,
        'deep': DeepThinkGenerator,
        'low': LowThinkGenerator,
        'adaptive': AdaptiveThinkGenerator,
        'minimal': MinimalThinkGenerator,
    }
    generator_cls = generators.get(generator_type, DefaultBranchesGenerator)
    return generator_cls(config, sandbox, abort_event, global_rules)
