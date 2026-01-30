import json
import os
from datetime import datetime
from loguru import logger

from gcri.tools.docker_sandbox import get_sandbox


class SandboxManager:
    def __init__(self, config):
        self.config = config
        self._project_dir = config.project_dir
        self._run_dir = config.run_dir
        self._work_dir = None
        self._log_dir = None
        self._docker_sandbox = None
        self._branch_containers = {}
        os.makedirs(self.run_dir, exist_ok=True)

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
    def docker_sandbox(self):
        if self._docker_sandbox is None:
            self._docker_sandbox = get_sandbox(self.config)
        return self._docker_sandbox

    def setup(self):
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._work_dir = os.path.join(self.run_dir, f'run-{timestamp}')
        self._log_dir = os.path.join(self.work_dir, 'logs')
        logger.info(f'📦 Creating sandbox run at: {self.work_dir}')
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def setup_branch(self, iteration_count, branch_index):
        container_id = self.docker_sandbox.setup_branch(
            iteration_count, branch_index, self.project_dir
        )
        self._branch_containers[(iteration_count, branch_index)] = container_id
        return container_id

    def get_branch_context(self, iteration_count, num_results):
        file_contexts = []
        for i in range(num_results):
            container_id = self._branch_containers.get((iteration_count, i))
            if container_id:
                file_contexts.append(f'- Branch {i}: Docker container {container_id[:12]}')
            else:
                file_contexts.append(f'- Branch {i}: (Container not found)')
        return '\n'.join(file_contexts)

    def save_iteration_log(self, index, result_data):
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f'log_iteration_{index:02d}.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        logger.info(f'Result of iteration {index+1} saved to: {log_path}')
        return log_path

    def get_winning_branch_path(self, index, branch_index):
        return self._branch_containers.get((index, branch_index))

    def commit_winning_branch(self, container_id):
        if not container_id:
            logger.warning('No container ID provided for commit.')
            return
        self.docker_sandbox.commit_to_host(container_id, self.project_dir)
        self.docker_sandbox.clean_up_container(container_id)

    def setup_verification_branches(self, iteration_count, aggregated_branches, source_containers):
        """
        Setup containers for verification branches after aggregation.

        Args:
            iteration_count: Current iteration index.
            aggregated_branches: List of AggregatedBranch from aggregator.
            source_containers: Dict mapping original branch index to container ID.

        Returns:
            Dict mapping verification branch index to container ID.
        """
        verification_containers = {}
        used_source_containers = set()

        for branch in aggregated_branches:
            if len(branch.source_indices) == 1:
                # Single source: reuse existing container
                src_idx = branch.source_indices[0]
                src_container = source_containers.get(src_idx)
                if src_container:
                    verification_containers[branch.index] = src_container
                    used_source_containers.add(src_idx)
                    logger.debug(
                        f'🔄 Verification branch {branch.index} reusing container from branch {src_idx}'
                    )
                else:
                    logger.warning(f'Source container for branch {src_idx} not found')
            else:
                # Multiple sources: create merged container
                containers_to_merge = []
                for src_idx in branch.source_indices:
                    src_container = source_containers.get(src_idx)
                    if src_container:
                        containers_to_merge.append(src_container)
                        used_source_containers.add(src_idx)

                if containers_to_merge:
                    try:
                        merged_container = self.docker_sandbox.merge_containers(
                            containers_to_merge, self.project_dir
                        )
                        verification_containers[branch.index] = merged_container
                        logger.info(
                            f'🔀 Verification branch {branch.index} merged from sources {branch.source_indices}'
                        )
                    except Exception as e:
                        logger.error(f'Failed to merge containers for branch {branch.index}: {e}')
                        # Fallback to first source
                        if containers_to_merge:
                            verification_containers[branch.index] = containers_to_merge[0]
                else:
                    logger.warning(f'No source containers found for branch {branch.index}')

        # Clean up unused source containers (discarded branches)
        for src_idx, container_id in source_containers.items():
            if src_idx not in used_source_containers:
                logger.debug(f'🗑️ Cleaning up discarded branch {src_idx} container')
                self.docker_sandbox.clean_up_container(container_id)
                # Remove from tracking
                key = (iteration_count, src_idx)
                if key in self._branch_containers:
                    del self._branch_containers[key]

        # Store verification containers with a different prefix to avoid conflicts
        for v_idx, container_id in verification_containers.items():
            self._branch_containers[(iteration_count, f'v_{v_idx}')] = container_id

        return verification_containers

    def get_verification_context(self, iteration_count, aggregated_branches):
        """
        Get file context for verification branches.

        Args:
            iteration_count: Current iteration index.
            aggregated_branches: List of AggregatedBranch.

        Returns:
            Formatted string with container information.
        """
        file_contexts = []
        for branch in aggregated_branches:
            container_id = self._branch_containers.get((iteration_count, f'v_{branch.index}'))
            sources = ', '.join(map(str, branch.source_indices))
            if container_id:
                file_contexts.append(
                    f'- Verification Branch {branch.index} (from [{sources}]): '
                    f'Container {container_id[:12]}'
                )
            else:
                file_contexts.append(
                    f'- Verification Branch {branch.index} (from [{sources}]): (Container not found)'
                )
        return '\n'.join(file_contexts)

    def clean_up(self):
        if self._docker_sandbox:
            self._docker_sandbox.clean_up_all()
        self._branch_containers.clear()

    def get_branch_files(self, iteration_count: int) -> dict:
        """
        Get files created/modified by each branch in this iteration.

        Args:
            iteration_count: Current iteration index.

        Returns:
            Dict mapping branch_index to dict of {file_path: content}.
        """
        branch_files = {}
        for (iter_idx, branch_idx), container_id in self._branch_containers.items():
            if iter_idx != iteration_count:
                continue
            if isinstance(branch_idx, str) and branch_idx.startswith('v_'):
                continue
            files = self.docker_sandbox.get_container_files(container_id)
            if files:
                branch_files[branch_idx] = files
                logger.debug(f'📁 Branch {branch_idx}: {len(files)} files collected')
        return branch_files

    def create_base_sandbox(self, source_indices: list, iteration_count: int) -> tuple:
        """
        Create merged base sandbox from selected branches.

        Args:
            source_indices: List of branch indices to merge.
            iteration_count: Current iteration index.

        Returns:
            Tuple of (container_id, file_summary_string).
        """
        if not source_indices:
            logger.warning('No source indices provided for base sandbox.')
            return None, ''

        source_containers = []
        for idx in source_indices:
            container_id = self._branch_containers.get((iteration_count, idx))
            if container_id:
                source_containers.append(container_id)

        if not source_containers:
            logger.warning('No containers found for specified indices.')
            return None, ''

        try:
            if len(source_containers) == 1:
                # Single source: clone it to preserve for next iteration
                base_container = self.docker_sandbox.clone_container(
                    source_containers[0], iteration_count+1, 'base'
                )
            else:
                base_container = self.docker_sandbox.merge_containers(
                    source_containers, self.project_dir
                )

            # Collect file summary
            files = self.docker_sandbox.get_container_files(base_container)
            file_list = list(files.keys()) if files else []
            summary = f'{len(file_list)} files from branches {source_indices}'
            if file_list:
                summary += f': {", ".join(file_list[:5])}'
                if len(file_list) > 5:
                    summary += f' ... and {len(file_list)-5} more'

            logger.info(f'🏗️ Created base sandbox {base_container[:12]} from branches {source_indices}')
            return base_container, summary

        except Exception as e:
            logger.error(f'Failed to create base sandbox: {e}')
            return None, ''

    def setup_branch_from_base(self, iteration_count: int, branch_index: int, base_container_id: str) -> str:
        """
        Setup branch by cloning base sandbox instead of project_dir.

        Args:
            iteration_count: Current iteration index.
            branch_index: Branch index.
            base_container_id: Container ID to clone from.

        Returns:
            New container ID for this branch.
        """
        try:
            container_id = self.docker_sandbox.clone_container(
                base_container_id, iteration_count, branch_index
            )
            self._branch_containers[(iteration_count, branch_index)] = container_id
            logger.debug(f'🔄 Branch {branch_index} cloned from base sandbox')
            return container_id
        except Exception as e:
            logger.error(f'Failed to setup branch from base: {e}. Falling back to fresh setup.')
            return self.setup_branch(iteration_count, branch_index)

