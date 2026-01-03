import json
import os
from datetime import datetime
from loguru import logger

from gcri.tools.docker_sandbox import get_sandbox


class SandboxManager:
    """
    Manages isolated Docker sandbox environments for GCRI branch execution.

    Creates Docker containers for each branch with complete isolation.
    Handles merging winning branch results back to the project directory via docker cp.
    """

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
        """Initialize a new run with timestamped work directory."""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._work_dir = os.path.join(self.run_dir, f'run-{timestamp}')
        self._log_dir = os.path.join(self.work_dir, 'logs')
        logger.info(f'📦 Creating sandbox run at: {self.work_dir}')
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def setup_branch(self, iteration_count, branch_index):
        """
        Create an isolated Docker container for a specific branch.

        Returns:
            str: Container ID for the branch.
        """
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
        """Returns container ID for the winning branch."""
        return self._branch_containers.get((index, branch_index))

    def commit_winning_branch(self, container_id):
        """
        Copy files from winning branch container to project directory.

        Args:
            container_id: Docker container ID of the winning branch.
        """
        if not container_id:
            logger.warning('No container ID provided for commit.')
            return
        self.docker_sandbox.commit_to_host(container_id, self.project_dir)
        self.docker_sandbox.cleanup_container(container_id)

    def cleanup(self):
        """Cleanup all Docker containers."""
        if self._docker_sandbox:
            self._docker_sandbox.cleanup_all()
        self._branch_containers.clear()
