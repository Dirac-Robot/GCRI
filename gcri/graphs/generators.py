"""
BranchesGenerator module for GCRI.

Provides interchangeable branch generation strategies that produce
RawHypothesis outputs from strategies. The default implementation
mirrors the original hypothesis + reasoning_and_refine workflow.
"""
from typing import Protocol, List, TYPE_CHECKING

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send
from loguru import logger

from gcri.graphs.schemas import Hypothesis, Reasoning, RawHypothesis, Strategies
from gcri.graphs.states import TaskState, BranchState
from gcri.tools.cli import build_model

if TYPE_CHECKING:
    from gcri.graphs.gcri_unit import GCRI


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


class DefaultBranchesGenerator:
    """
    Default branch generator that mirrors the original GCRI workflow.

    Executes: sample_strategies → parallel(hypothesis + reasoning_and_refine)
    """

    def __init__(self, gcri: 'GCRI'):
        self.gcri = gcri
        self.config = gcri.config
        self.sandbox = gcri.sandbox
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal subgraph for hypothesis generation."""
        branch = StateGraph(BranchState)
        branch.add_node('sample_hypothesis', self._sample_hypothesis)
        branch.add_node('reasoning_and_refine', self._reasoning_and_refine)
        branch.add_edge(START, 'sample_hypothesis')
        branch.add_edge('sample_hypothesis', 'reasoning_and_refine')
        branch.add_edge('reasoning_and_refine', END)
        self._branch_workflow = branch.compile()

    def _sample_hypothesis(self, state: BranchState):
        """Generate initial hypothesis for a branch."""
        self.gcri._check_abort()
        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=state.index,
            data={'type': 'processing'}
        ).info(f'Iter #{state.count_in_branch+1} | Branch[{state.index}] Sampling hypothesis...')

        hypothesis_config = self.config.agents.branches[state.index].hypothesis
        agent = build_model(
            hypothesis_config.model_id,
            hypothesis_config.get('gcri_options'),
            container_id=state.container_id,
            **hypothesis_config.parameters
        ).with_structured_output(schema=Hypothesis)

        template = self.gcri._load_template_with_rules(
            self.config.templates.hypothesis,
            task=state.task_in_branch,
            strictness=state.strictness,
            strategy=state.strategy,
            intent_analysis=state.intent_analysis_in_branch
        )
        hypothesis = self.gcri._invoke_with_retry(
            agent, template, f'Hypothesis agent at strategy #{state.index+1}'
        )

        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=state.index,
            data={'hypothesis': hypothesis.hypothesis, 'container_id': state.container_id}
        ).info(f'Iter #{state.count_in_branch+1} | Branch[{state.index}] Hypothesis: {hypothesis.hypothesis[:80]}...')

        return {'hypothesis': hypothesis.hypothesis}

    def _reasoning_and_refine(self, state: BranchState):
        """Apply reasoning to refine the hypothesis."""
        self.gcri._check_abort()
        logger.bind(
            ui_event='node_update',
            node='reasoning',
            branch=state.index,
            data={'type': 'processing'}
        ).info(f'Iter #{state.count_in_branch+1} | Branch[{state.index}] Reasoning...')

        reasoning_config = self.config.agents.branches[state.index].reasoning
        agent = build_model(
            reasoning_config.model_id,
            reasoning_config.get('gcri_options'),
            container_id=state.container_id,
            **reasoning_config.parameters
        ).with_structured_output(schema=Reasoning)

        template = self.gcri._load_template_with_rules(
            self.config.templates.reasoning,
            task=state.task_in_branch,
            strategy=state.strategy,
            hypothesis=state.hypothesis,
            intent_analysis=state.intent_analysis_in_branch
        )
        reasoning = self.gcri._invoke_with_retry(
            agent, template, f'Reasoning agent at hypothesis #{state.index+1}'
        )

        # Build RawHypothesis output
        raw_hyp = RawHypothesis(
            index=state.index,
            strategy_name=state.strategy.name,
            strategy_description=state.strategy.description,
            hypothesis=reasoning.refined_hypothesis,
            reasoning=reasoning.reasoning,
            container_id=state.container_id
        )

        logger.bind(
            ui_event='node_update',
            node='reasoning',
            branch=state.index,
            data={
                'reasoning': reasoning.reasoning,
                'hypothesis': reasoning.refined_hypothesis,
                'container_id': state.container_id
            }
        ).info(f'Iter #{state.count_in_branch+1} | Branch[{state.index}] Refined: {reasoning.refined_hypothesis[:80]}...')

        return {
            'reasoning': reasoning.reasoning,
            'hypothesis': reasoning.refined_hypothesis,
            'raw_hypothesis_output': raw_hyp
        }

    def _map_branches(self, state: TaskState):
        """Map strategies to parallel branch executions."""
        logger.bind(ui_event='phase_change', phase='hypothesis_generation').info(
            'Starting Hypothesis Generation...'
        )
        num_branches = min(len(self.config.agents.branches), len(state.strategies))
        logger.info(f'🌿 Spawning {num_branches} hypothesis branches...')

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

    def _collect_raw_hypotheses(self, state: TaskState):
        """Collect RawHypothesis outputs from all branches."""
        # raw_hypotheses is populated via operator.add from BranchState.raw_hypothesis_output
        # But we need to extract from results since BranchState uses 'results'
        # Actually, we need to access raw_hypothesis_output from BranchState
        pass

    def generate(self, state: TaskState) -> dict:
        """Generate strategies and hypotheses."""
        # Step 1: Sample strategies
        strategies_result = self.gcri.sample_strategies(state)
        updated_state = state.model_copy(update=strategies_result)

        # Step 2: Execute branch workflow
        graph = StateGraph(TaskState)
        graph.add_node('branch_executor', self._branch_workflow)
        graph.add_conditional_edges(START, self._map_branches, ['branch_executor'])
        graph.add_edge('branch_executor', END)
        workflow = graph.compile()

        result = workflow.invoke(updated_state.model_dump())

        # Extract raw_hypotheses from branch results
        raw_hypotheses = []
        for branch_result in result.get('results', []):
            if hasattr(branch_result, 'raw_hypothesis_output') and branch_result.raw_hypothesis_output:
                raw_hypotheses.append(branch_result.raw_hypothesis_output)

        return {
            'strategies': strategies_result['strategies'],
            'intent_analysis': strategies_result['intent_analysis'],
            'task_strictness': strategies_result['task_strictness'],
            'raw_hypotheses': raw_hypotheses
        }


class DeepThinkGenerator:
    """
    Deep thinking branch generator with full refinement workflow.

    Executes: sample_strategies → parallel(hypothesis + reasoning_and_refine)

    This is equivalent to the DefaultBranchesGenerator, providing the
    standard GCRI workflow with strategy sampling, hypothesis generation,
    and reasoning-based refinement.
    """

    def __init__(self, gcri: 'GCRI'):
        self.gcri = gcri
        self.config = gcri.config
        self.sandbox = gcri.sandbox
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal subgraph for hypothesis generation with refine."""
        branch = StateGraph(BranchState)
        branch.add_node('sample_hypothesis', self._sample_hypothesis)
        branch.add_node('reasoning_and_refine', self._reasoning_and_refine)
        branch.add_edge(START, 'sample_hypothesis')
        branch.add_edge('sample_hypothesis', 'reasoning_and_refine')
        branch.add_edge('reasoning_and_refine', END)
        self._branch_workflow = branch.compile()

    def _sample_hypothesis(self, state: BranchState):
        """Generate initial hypothesis for a branch."""
        self.gcri._check_abort()
        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=state.index,
            data={'type': 'processing'}
        ).info(f'[DeepThink] Iter #{state.count_in_branch+1} | Branch[{state.index}] Sampling hypothesis...')

        hypothesis_config = self.config.agents.branches[state.index].hypothesis
        agent = build_model(
            hypothesis_config.model_id,
            hypothesis_config.get('gcri_options'),
            container_id=state.container_id,
            **hypothesis_config.parameters
        ).with_structured_output(schema=Hypothesis)

        template = self.gcri._load_template_with_rules(
            self.config.templates.hypothesis,
            task=state.task_in_branch,
            strictness=state.strictness,
            strategy=state.strategy,
            intent_analysis=state.intent_analysis_in_branch
        )
        hypothesis = self.gcri._invoke_with_retry(
            agent, template, f'Hypothesis agent at strategy #{state.index+1}'
        )

        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=state.index,
            data={'hypothesis': hypothesis.hypothesis, 'container_id': state.container_id}
        ).info(f'[DeepThink] Iter #{state.count_in_branch+1} | Branch[{state.index}] Hypothesis: {hypothesis.hypothesis[:80]}...')

        return {'hypothesis': hypothesis.hypothesis}

    def _reasoning_and_refine(self, state: BranchState):
        """Apply reasoning to refine the hypothesis."""
        self.gcri._check_abort()
        logger.bind(
            ui_event='node_update',
            node='reasoning',
            branch=state.index,
            data={'type': 'processing'}
        ).info(f'[DeepThink] Iter #{state.count_in_branch+1} | Branch[{state.index}] Reasoning and refining...')

        reasoning_config = self.config.agents.branches[state.index].reasoning
        agent = build_model(
            reasoning_config.model_id,
            reasoning_config.get('gcri_options'),
            container_id=state.container_id,
            **reasoning_config.parameters
        ).with_structured_output(schema=Reasoning)

        template = self.gcri._load_template_with_rules(
            self.config.templates.reasoning,
            task=state.task_in_branch,
            strategy=state.strategy,
            hypothesis=state.hypothesis,
            intent_analysis=state.intent_analysis_in_branch
        )
        reasoning = self.gcri._invoke_with_retry(
            agent, template, f'Reasoning agent at hypothesis #{state.index+1}'
        )

        raw_hyp = RawHypothesis(
            index=state.index,
            strategy_name=state.strategy.name,
            strategy_description=state.strategy.description,
            hypothesis=reasoning.refined_hypothesis,
            reasoning=reasoning.reasoning,
            container_id=state.container_id
        )

        logger.bind(
            ui_event='node_update',
            node='reasoning',
            branch=state.index,
            data={
                'reasoning': reasoning.reasoning,
                'hypothesis': reasoning.refined_hypothesis,
                'container_id': state.container_id
            }
        ).info(f'[DeepThink] Iter #{state.count_in_branch+1} | Branch[{state.index}] Refined: {reasoning.refined_hypothesis[:80]}...')

        return {
            'reasoning': reasoning.reasoning,
            'hypothesis': reasoning.refined_hypothesis,
            'raw_hypothesis_output': raw_hyp
        }

    def _map_branches(self, state: TaskState):
        """Map strategies to parallel branch executions."""
        logger.bind(ui_event='phase_change', phase='hypothesis_generation').info(
            '[DeepThink] Starting Hypothesis Generation...'
        )
        num_branches = min(len(self.config.agents.branches), len(state.strategies))
        logger.info(f'🌿 [DeepThink] Spawning {num_branches} hypothesis branches...')

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
        """Generate strategies and hypotheses with full refinement."""
        strategies_result = self.gcri.sample_strategies(state)
        updated_state = state.model_copy(update=strategies_result)

        graph = StateGraph(TaskState)
        graph.add_node('branch_executor', self._branch_workflow)
        graph.add_conditional_edges(START, self._map_branches, ['branch_executor'])
        graph.add_edge('branch_executor', END)
        workflow = graph.compile()

        result = workflow.invoke(updated_state.model_dump())

        raw_hypotheses = []
        for branch_result in result.get('results', []):
            if hasattr(branch_result, 'raw_hypothesis_output') and branch_result.raw_hypothesis_output:
                raw_hypotheses.append(branch_result.raw_hypothesis_output)

        return {
            'strategies': strategies_result['strategies'],
            'intent_analysis': strategies_result['intent_analysis'],
            'task_strictness': strategies_result['task_strictness'],
            'raw_hypotheses': raw_hypotheses
        }


class LowThinkGenerator:
    """
    Low thinking branch generator for fast hypothesis generation.

    Executes: sample_strategies → parallel(hypothesis only)

    Skips the reasoning/refine step for faster execution. The raw hypothesis
    is used directly without refinement.
    """

    def __init__(self, gcri: 'GCRI'):
        self.gcri = gcri
        self.config = gcri.config
        self.sandbox = gcri.sandbox
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal subgraph for hypothesis-only generation."""
        branch = StateGraph(BranchState)
        branch.add_node('sample_hypothesis', self._sample_hypothesis)
        branch.add_edge(START, 'sample_hypothesis')
        branch.add_edge('sample_hypothesis', END)
        self._branch_workflow = branch.compile()

    def _sample_hypothesis(self, state: BranchState):
        """Generate hypothesis and directly output as RawHypothesis (no refine)."""
        self.gcri._check_abort()
        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=state.index,
            data={'type': 'processing'}
        ).info(f'[LowThink] Iter #{state.count_in_branch+1} | Branch[{state.index}] Sampling hypothesis (no refine)...')

        hypothesis_config = self.config.agents.branches[state.index].hypothesis
        agent = build_model(
            hypothesis_config.model_id,
            hypothesis_config.get('gcri_options'),
            container_id=state.container_id,
            **hypothesis_config.parameters
        ).with_structured_output(schema=Hypothesis)

        template = self.gcri._load_template_with_rules(
            self.config.templates.hypothesis,
            task=state.task_in_branch,
            strictness=state.strictness,
            strategy=state.strategy,
            intent_analysis=state.intent_analysis_in_branch
        )
        hypothesis = self.gcri._invoke_with_retry(
            agent, template, f'Hypothesis agent at strategy #{state.index+1}'
        )

        # Directly create RawHypothesis without reasoning step
        raw_hyp = RawHypothesis(
            index=state.index,
            strategy_name=state.strategy.name,
            strategy_description=state.strategy.description,
            hypothesis=hypothesis.hypothesis,
            reasoning='[LowThink: No refinement applied]',
            container_id=state.container_id
        )

        logger.bind(
            ui_event='node_update',
            node='hypothesis',
            branch=state.index,
            data={'hypothesis': hypothesis.hypothesis, 'container_id': state.container_id}
        ).info(f'[LowThink] Iter #{state.count_in_branch+1} | Branch[{state.index}] Hypothesis: {hypothesis.hypothesis[:80]}...')

        return {
            'hypothesis': hypothesis.hypothesis,
            'raw_hypothesis_output': raw_hyp
        }

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
        strategies_result = self.gcri.sample_strategies(state)
        updated_state = state.model_copy(update=strategies_result)

        graph = StateGraph(TaskState)
        graph.add_node('branch_executor', self._branch_workflow)
        graph.add_conditional_edges(START, self._map_branches, ['branch_executor'])
        graph.add_edge('branch_executor', END)
        workflow = graph.compile()

        result = workflow.invoke(updated_state.model_dump())

        raw_hypotheses = []
        for branch_result in result.get('results', []):
            if hasattr(branch_result, 'raw_hypothesis_output') and branch_result.raw_hypothesis_output:
                raw_hypotheses.append(branch_result.raw_hypothesis_output)

        return {
            'strategies': strategies_result['strategies'],
            'intent_analysis': strategies_result['intent_analysis'],
            'task_strictness': strategies_result['task_strictness'],
            'raw_hypotheses': raw_hypotheses
        }


def get_branches_generator(generator_type: str, gcri: 'GCRI'):
    """Factory function to get the appropriate BranchesGenerator.

    Args:
        generator_type: One of 'default', 'deep', 'low'
        gcri: GCRI instance

    Returns:
        BranchesGenerator instance
    """
    generators = {
        'default': DefaultBranchesGenerator,
        'deep': DeepThinkGenerator,
        'low': LowThinkGenerator,
    }
    generator_cls = generators.get(generator_type, DefaultBranchesGenerator)
    return generator_cls(gcri)
