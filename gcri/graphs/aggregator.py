"""
HypothesisAggregator module for GCRI.

Aggregates multiple RawHypothesis outputs into optimized combinations,
reducing redundancy and identifying complementary approaches.
"""
from typing import List

from loguru import logger

from gcri.graphs.schemas import RawHypothesis, AggregationResult, AggregatedBranch
from gcri.graphs.states import TaskState
from gcri.tools.cli import build_model


class HypothesisAggregator:
    """
    Aggregates multiple hypotheses into optimal combinations.

    Takes RawHypothesis list from BranchesGenerator and produces
    AggregationResult with merged/filtered branches for verification.
    """

    def __init__(self, config):
        self.config = config
        self._agent = None

    @property
    def agent(self):
        """Lazy initialization of aggregator agent."""
        if self._agent is None:
            aggregator_config = self.config.agents.aggregator
            self._agent = build_model(
                aggregator_config.model_id,
                aggregator_config.get('gcri_options'),
                **aggregator_config.parameters
            ).with_structured_output(schema=AggregationResult)
        return self._agent

    def _build_template(self, raw_hypotheses: List[RawHypothesis], task: str) -> str:
        """Build prompt template for aggregation."""
        hypotheses_text = []
        for hyp in raw_hypotheses:
            hypotheses_text.append(
                f'[Branch {hyp.index}] Strategy: {hyp.strategy_name}\n'
                f'Description: {hyp.strategy_description}\n'
                f'Hypothesis: {hyp.hypothesis}\n'
                f'Reasoning: {hyp.reasoning}\n'
            )

        max_branches = self.config.aggregation.get('max_output_branches', 3)

        template_path = self.config.templates.aggregator
        with open(template_path, 'r') as f:
            template = f.read()

        return template.format(
            task=task,
            raw_hypotheses='\n---\n'.join(hypotheses_text),
            max_branches=max_branches
        )

    def aggregate(self, state: TaskState) -> AggregationResult:
        """
        Aggregate raw hypotheses into optimized combinations.

        Args:
            state: TaskState with raw_hypotheses populated.

        Returns:
            AggregationResult with merged branches.
        """
        raw_hypotheses = state.raw_hypotheses

        if not raw_hypotheses:
            logger.warning('No raw hypotheses to aggregate.')
            return AggregationResult(
                branches=[],
                discarded_indices=[],
                aggregation_summary='No hypotheses provided for aggregation.'
            )

        logger.bind(
            ui_event='phase_change',
            phase='aggregation'
        ).info('Starting Hypothesis Aggregation...')

        logger.info(f'📊 Aggregating {len(raw_hypotheses)} hypotheses...')

        # Check if passthrough is enabled for single-source branches
        allow_passthrough = self.config.aggregation.get('allow_single_source_passthrough', True)

        # If only one hypothesis, pass through directly
        if len(raw_hypotheses) == 1 and allow_passthrough:
            hyp = raw_hypotheses[0]
            logger.info('Single hypothesis detected, passing through directly.')
            return AggregationResult(
                branches=[
                    AggregatedBranch(
                        index=0,
                        combined_hypothesis=hyp.hypothesis,
                        source_indices=[hyp.index],
                        merge_reasoning='Single hypothesis passthrough'
                    )
                ],
                discarded_indices=[],
                aggregation_summary='Single hypothesis passed through without aggregation.'
            )

        # Build and invoke aggregation
        template = self._build_template(raw_hypotheses, state.task)
        result = self.agent.invoke(template)

        # Log aggregation results
        logger.info(f'📊 Aggregation complete: {len(result.branches)} branches output')
        for branch in result.branches:
            sources = ', '.join(map(str, branch.source_indices))
            logger.info(f'  → Branch {branch.index}: merged from [{sources}]')

        if result.discarded_indices:
            logger.info(f'  ⚠️ Discarded branches: {result.discarded_indices}')

        logger.bind(
            ui_event='node_update',
            node='aggregation',
            data={
                'input_count': len(raw_hypotheses),
                'output_count': len(result.branches),
                'discarded': result.discarded_indices
            }
        ).info('Aggregation completed.')

        return result

    def aggregate_node(self, state: TaskState) -> dict:
        """
        Node function for LangGraph integration.

        Args:
            state: TaskState with raw_hypotheses populated.

        Returns:
            dict with 'aggregated_branches' key.
        """
        result = self.aggregate(state)
        return {
            'aggregated_branches': result.branches
        }
