"""
HypothesisAggregator module for GCRI.

Aggregates multiple RawHypothesis outputs into optimized combinations,
performing intelligent code-level merging when branches modify files.
"""
import uuid
from typing import List, Dict

from loguru import logger

from gcri.graphs.schemas import RawHypothesis, AggregationResult, AggregatedBranch
from gcri.graphs.states import TaskState
from gcri.tools.cli import build_model, CLI_TOOLS, GlobalVariables


class HypothesisAggregator:
    """
    Aggregates multiple hypotheses into optimal combinations.

    Takes RawHypothesis list from BranchesGenerator and produces
    AggregationResult with merged/filtered branches for verification.
    Performs intelligent file-level merging via LLM agents.
    """

    def __init__(self, config, sandbox):
        self.config = config
        self.sandbox = sandbox
        self._agent = None

    @property
    def agent(self):
        """Lazy initialization of aggregator agent with code tools."""
        if self._agent is None:
            aggregator_config = self.config.agents.aggregator
            self._agent = build_model(
                aggregator_config.model_id,
                aggregator_config.get('gcri_options'),
                **aggregator_config.parameters
            ).with_structured_output(schema=AggregationResult)
        return self._agent

    def _get_container_files(self, container_id: str) -> Dict[str, str]:
        """
        Get files that were created/modified by LLM in the container.

        Uses .gcri_baseline marker file created at container setup time
        to identify only files newer than baseline.
        """
        files = {}
        try:
            result = self.sandbox._execute_in_container(
                container_id,
                ['sh', '-c',
                 'if [ -f /workspace/.gcri_baseline ]; then '
                 'find /workspace -type f -newer /workspace/.gcri_baseline '
                 '-not -path "*/__pycache__/*" '
                 '-not -path "*/.git/*" '
                 '-not -path "*/.venv/*" '
                 '-not -path "*/.gcri*" '
                 '-not -name "*.pyc" '
                 '-not -name ".gcri_baseline"; '
                 'fi'],
                stdout_only=True
            )
            if result.startswith('Error'):
                logger.warning(f'Failed to list modified files: {result}')
                return files

            file_paths = [p.strip() for p in result.strip().split('\n') if p.strip()]
            logger.debug(f'Found {len(file_paths)} files modified since baseline')

            for file_path in file_paths:
                content = self.sandbox._execute_in_container(
                    container_id,
                    ['cat', file_path],
                    stdout_only=True
                )
                if not content.startswith('Error (exit'):
                    rel_path = file_path.replace('/workspace/', '')
                    files[rel_path] = content

        except Exception as e:
            logger.warning(f'Error getting container files: {e}')

        return files

    def _build_template_with_files(
        self,
        raw_hypotheses: List[RawHypothesis],
        task: str,
        branch_files: Dict[int, Dict[str, str]]
    ) -> str:
        """Build prompt template including file contents from each branch."""
        hypotheses_text = []
        for hyp in raw_hypotheses:
            files_info = branch_files.get(hyp.index, {})
            files_section = ''
            if files_info:
                file_details = []
                for path, content in files_info.items():
                    file_details.append(f'  📄 {path}:\n```\n{content}\n```')
                files_section = '\nFiles Modified:\n' + '\n'.join(file_details)

            hypotheses_text.append(
                f'[Branch {hyp.index}] Strategy: {hyp.strategy_name}\n'
                f'Description: {hyp.strategy_description}\n'
                f'Hypothesis: {hyp.hypothesis}\n'
                f'Reasoning: {hyp.reasoning}'
                f'{files_section}\n'
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

    def _create_merged_container(
        self,
        source_indices: List[int],
        branch_containers: Dict[int, str]
    ) -> str:
        """Create a new container with merged files from source branches."""
        source_containers = [
            branch_containers[idx] for idx in source_indices
            if idx in branch_containers
        ]

        if not source_containers:
            logger.warning('No source containers found for merge.')
            return ''

        if len(source_containers) == 1:
            # Single source, reuse container
            return source_containers[0]

        # Multiple sources: use docker_sandbox merge
        try:
            merged_container = self.sandbox.merge_containers(
                source_containers, self.config.project_dir
            )
            return merged_container
        except Exception as e:
            logger.error(f'Failed to merge containers: {e}')
            # Fallback to first source
            return source_containers[0] if source_containers else ''

    def _execute_file_merge(
        self,
        aggregation_result: AggregationResult,
        branch_containers: Dict[int, str],
        branch_files: Dict[int, Dict[str, str]],
        task: str
    ) -> AggregationResult:
        """Execute intelligent file merging for each aggregated branch."""
        updated_branches = []

        for branch in aggregation_result.branches:
            if len(branch.source_indices) == 1:
                # Single source: use its container directly
                src_idx = branch.source_indices[0]
                container_id = branch_containers.get(src_idx, '')
                updated_branches.append(AggregatedBranch(
                    index=branch.index,
                    combined_hypothesis=branch.combined_hypothesis,
                    source_indices=branch.source_indices,
                    merge_reasoning=branch.merge_reasoning,
                    container_id=container_id
                ))
            else:
                # Multiple sources: need intelligent merge
                source_files = {
                    idx: branch_files.get(idx, {})
                    for idx in branch.source_indices
                }

                # Check if files actually need merging (same path modified by multiple branches)
                all_paths = set()
                conflict_paths = set()
                for idx, files in source_files.items():
                    for path in files.keys():
                        if path in all_paths:
                            conflict_paths.add(path)
                        all_paths.add(path)

                if conflict_paths:
                    # Real conflicts exist - need LLM to merge
                    logger.info(f'🔀 Branch {branch.index}: {len(conflict_paths)} file conflicts detected')
                    merged_container = self._intelligent_merge(
                        branch, source_files, branch_containers, task
                    )
                else:
                    # No conflicts - just combine containers physically
                    merged_container = self._create_merged_container(
                        branch.source_indices, branch_containers
                    )

                updated_branches.append(AggregatedBranch(
                    index=branch.index,
                    combined_hypothesis=branch.combined_hypothesis,
                    source_indices=branch.source_indices,
                    merge_reasoning=branch.merge_reasoning,
                    container_id=merged_container
                ))

        return AggregationResult(
            branches=updated_branches,
            discarded_indices=aggregation_result.discarded_indices,
            aggregation_summary=aggregation_result.aggregation_summary
        )

    def _intelligent_merge(
        self,
        branch: AggregatedBranch,
        source_files: Dict[int, Dict[str, str]],
        branch_containers: Dict[int, str],
        task: str
    ) -> str:
        """Use LLM to intelligently merge conflicting files."""
        # First create base merged container
        base_container = self._create_merged_container(
            branch.source_indices, branch_containers
        )

        # Build merge prompt for conflicting files
        all_paths = set()
        for files in source_files.values():
            all_paths.update(files.keys())

        conflict_details = []
        for path in all_paths:
            versions = []
            for idx, files in source_files.items():
                if path in files:
                    content = files[path]
                    versions.append(f'### Branch {idx} version:\n```\n{content}\n```')

            if len(versions) > 1:
                conflict_details.append(f'## File: {path}\n' + '\n'.join(versions))

        if not conflict_details:
            return base_container

        merge_prompt = f"""You are merging code from multiple branches for this task:
{task}

The aggregated hypothesis is:
{branch.combined_hypothesis}

The following files have different versions from different branches.
Analyze each version and create a merged version that:
1. Combines the best parts from each branch
2. Resolves any conflicts intelligently
3. Maintains code consistency

{chr(10).join(conflict_details)}

Use the write_file tool to write the merged versions to the workspace.
After writing all merged files, respond with a summary of what was merged.
"""

        # Create agent with code tools for this container
        aggregator_config = self.config.agents.aggregator
        merge_agent = build_model(
            aggregator_config.model_id,
            aggregator_config.get('gcri_options'),
            container_id=base_container,
            **aggregator_config.parameters
        )

        # Execute merge (using raw invoke, not structured output)
        try:
            # Set container context
            token = GlobalVariables.CONTAINER_VAR.set(base_container)
            try:
                result = merge_agent.agent.bind_tools(CLI_TOOLS).invoke(merge_prompt)
                # Process tool calls if any
                if result.tool_calls:
                    for call in result.tool_calls:
                        name = call['name']
                        args = call['args']
                        if name == 'write_file':
                            # Execute file write
                            from gcri.tools.cli import write_file
                            write_file.invoke(args)
                            logger.debug(f'  Merged file: {args.get("file_path", "unknown")}')
            finally:
                GlobalVariables.CONTAINER_VAR.reset(token)

            logger.info(f'🔀 Intelligent merge completed for branch {branch.index}')
        except Exception as e:
            logger.error(f'Intelligent merge failed: {e}')

        return base_container

    def aggregate(self, state: TaskState, branch_containers: Dict[int, str] = None) -> AggregationResult:
        """
        Aggregate raw hypotheses into optimized combinations.

        Args:
            state: TaskState with raw_hypotheses populated.
            branch_containers: Mapping of branch index to container ID.

        Returns:
            AggregationResult with merged branches including container_ids.
        """
        raw_hypotheses = state.raw_hypotheses
        branch_containers = branch_containers or {}

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

        # Collect file contents from each branch container
        branch_files = {}
        for hyp in raw_hypotheses:
            container_id = branch_containers.get(hyp.index)
            if container_id:
                branch_files[hyp.index] = self._get_container_files(container_id)
                logger.debug(f'  Branch {hyp.index}: {len(branch_files[hyp.index])} files')

        # Check if passthrough is enabled for single-source branches
        allow_passthrough = self.config.aggregation.get('allow_single_source_passthrough', True)

        # If only one hypothesis, pass through directly
        if len(raw_hypotheses) == 1 and allow_passthrough:
            hyp = raw_hypotheses[0]
            container_id = branch_containers.get(hyp.index, '')
            logger.info('Single hypothesis detected, passing through directly.')
            return AggregationResult(
                branches=[
                    AggregatedBranch(
                        index=0,
                        combined_hypothesis=hyp.hypothesis,
                        source_indices=[hyp.index],
                        merge_reasoning='Single hypothesis passthrough',
                        container_id=container_id
                    )
                ],
                discarded_indices=[],
                aggregation_summary='Single hypothesis passed through without aggregation.'
            )

        # Build template with file information
        template = self._build_template_with_files(raw_hypotheses, state.task, branch_files)
        result = self.agent.invoke(template)

        # Handle None result - fallback to first branch passthrough
        if result is None:
            logger.warning('Aggregation LLM returned None, falling back to first branch passthrough')
            hyp = raw_hypotheses[0]
            container_id = branch_containers.get(hyp.index, '')
            return AggregationResult(
                branches=[
                    AggregatedBranch(
                        index=0,
                        combined_hypothesis=hyp.hypothesis,
                        source_indices=[hyp.index],
                        merge_reasoning='Aggregation fallback: LLM returned None',
                        container_id=container_id
                    )
                ],
                discarded_indices=[i for i in range(1, len(raw_hypotheses))],
                aggregation_summary='Fallback: first hypothesis passed through due to aggregation failure.'
            )

        # Log aggregation results
        logger.info(f'📊 Aggregation complete: {len(result.branches)} branches output')
        for branch in result.branches:
            sources = ', '.join(map(str, branch.source_indices))
            logger.info(f'  → Branch {branch.index}: merged from [{sources}]')

        if result.discarded_indices:
            logger.info(f'  ⚠️ Discarded branches: {result.discarded_indices}')

        # Execute file-level merging
        result = self._execute_file_merge(result, branch_containers, branch_files, state.task)

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
        # Extract container mapping from raw_hypotheses
        branch_containers = {
            hyp.index: hyp.container_id
            for hyp in state.raw_hypotheses
            if hyp.container_id
        }
        result = self.aggregate(state, branch_containers)
        return {
            'aggregated_branches': result.branches
        }
