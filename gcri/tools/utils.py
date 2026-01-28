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

        for branch in aggregated_branches:
            if len(branch.source_indices) == 1:
                # Single source: reuse existing container
                src_idx = branch.source_indices[0]
                src_container = source_containers.get(src_idx)
                if src_container:
                    verification_containers[branch.index] = src_container
                    logger.debug(
                        f'🔄 Verification branch {branch.index} reusing container from branch {src_idx}'
                    )
                else:
                    logger.warning(f'Source container for branch {src_idx} not found')
            else:
                # Multiple sources: create new merged container
                # For now, use the first source container (TODO: implement file merging)
                primary_src = branch.source_indices[0]
                src_container = source_containers.get(primary_src)
                if src_container:
                    verification_containers[branch.index] = src_container
                    logger.debug(
                        f'🔀 Verification branch {branch.index} using primary source {primary_src} '
                        f'(merged from {branch.source_indices})'
                    )
                else:
                    logger.warning(f'Primary source container for branch {primary_src} not found')

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

