import os
import shutil
import subprocess
import uuid

from loguru import logger


class DockerSandbox:
    """
    Executes code in fully isolated Docker containers.

    Uses docker cp for file transfer instead of volume mounts.
    Applies network isolation and resource limits.
    """

    def __init__(self, config):
        self.image = config.sandbox.image
        self.timeout = config.sandbox.timeout
        self.memory_limit = config.sandbox.memory_limit
        self.cpu_limit = config.sandbox.cpu_limit
        self.network_mode = config.sandbox.network_mode
        self._docker_available = None
        self._containers = {}

    @property
    def docker_available(self):
        if self._docker_available is None:
            try:
                result = subprocess.run(
                    ['docker', 'version'],
                    capture_output=True,
                    timeout=5
                )
                self._docker_available = result.returncode == 0
            except Exception:
                self._docker_available = False
        return self._docker_available

    def setup_branch(self, iteration: int, branch: int, source_dir: str) -> str:
        """
        Create isolated container for a branch and copy project files.

        Args:
            iteration: Current iteration index
            branch: Branch index
            source_dir: Project directory to copy

        Returns:
            container_id: Created container ID
        """
        if not self.docker_available:
            raise RuntimeError('Docker is not available on this system.')

        branch_id = f'{iteration}_{branch}'
        container_name = f'gcri_branch_{branch_id}_{uuid.uuid4().hex[:8]}'

        create_cmd = [
            'docker', 'create',
            '--name', container_name,
            f'--memory={self.memory_limit}',
            f'--cpus={self.cpu_limit}',
            f'--network={self.network_mode}',
            '-w', '/workspace',
            '-t',
            self.image,
            'tail', '-f', '/dev/null'
        ]

        try:
            result = subprocess.run(create_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f'Failed to create container: {result.stderr}')
            container_id = result.stdout.strip()

            subprocess.run(['docker', 'start', container_id], capture_output=True, timeout=10)

            self._copy_to_container(source_dir, container_id)
            self._sync_python_environment(source_dir, container_id)

            self._containers[branch_id] = container_id
            logger.debug(f'🐳 Docker container created: {container_id[:12]} for branch {branch_id}')
            return container_id
        except Exception as e:
            logger.error(f'Failed to setup Docker branch: {e}')
            raise

    def _copy_to_container(self, source_dir: str, container_id: str):
        """Copy project files to container with ignore patterns."""
        temp_dir = f'/tmp/gcri_copy_{uuid.uuid4().hex[:8]}'
        ignore_patterns = {'.git', '__pycache__', 'venv', 'env', 'node_modules', '.idea', '.vscode', '.gcri'}

        try:
            os.makedirs(temp_dir, exist_ok=True)
            for item in os.listdir(source_dir):
                if item in ignore_patterns:
                    continue
                src = os.path.join(source_dir, item)
                dst = os.path.join(temp_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*ignore_patterns))
                else:
                    shutil.copy2(src, dst)

            subprocess.run(
                ['docker', 'cp', f'{temp_dir}/.', f'{container_id}:/workspace'],
                capture_output=True,
                timeout=60
            )
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _sync_python_environment(self, source_dir: str, container_id: str):
        """Install Python dependencies in container if requirements file exists."""
        requirements_path = os.path.join(source_dir, 'requirements.txt')
        pyproject_path = os.path.join(source_dir, 'pyproject.toml')
        setup_path = os.path.join(source_dir, 'setup.py')

        if os.path.exists(requirements_path):
            logger.info('📦 Installing requirements.txt in container...')
            result = self._execute_in_container(
                container_id,
                ['pip', 'install', '-q', '-r', '/workspace/requirements.txt']
            )
            if 'Error' not in result:
                logger.debug('✅ Requirements installed successfully')
            else:
                logger.warning(f'⚠️ Some packages may have failed: {result[:200]}')
        elif os.path.exists(pyproject_path):
            logger.info('📦 Installing from pyproject.toml in container...')
            result = self._execute_in_container(
                container_id,
                ['pip', 'install', '-q', '-e', '/workspace']
            )
            if 'Error' not in result:
                logger.debug('✅ Project installed successfully')
            else:
                logger.warning(f'⚠️ Some packages may have failed: {result[:200]}')
        elif os.path.exists(setup_path):
            logger.info('📦 Installing from setup.py in container...')
            self._execute_in_container(
                container_id,
                ['pip', 'install', '-q', '-e', '/workspace']
            )

    def execute_command(self, container_id: str, command: str) -> str:
        """Execute shell command in container."""
        return self._execute_in_container(container_id, ['sh', '-c', command])

    def execute_python(self, container_id: str, code: str) -> str:
        """Execute Python code in container."""
        script_name = f'_script_{uuid.uuid4().hex[:8]}.py'
        write_cmd = f"cat > /workspace/{script_name} << 'GCRI_EOF'\n{code}\nGCRI_EOF"
        self._execute_in_container(container_id, ['sh', '-c', write_cmd])
        result = self._execute_in_container(container_id, ['python', script_name])
        self._execute_in_container(container_id, ['rm', '-f', script_name])
        return result

    def _execute_in_container(self, container_id: str, command: list) -> str:
        """Execute command via docker exec."""
        exec_cmd = ['docker', 'exec', container_id]+command

        try:
            result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout+5
            )
            output = result.stdout+result.stderr
            if result.returncode != 0 and not output.strip():
                output = f'Exit Code {result.returncode}'
            return output if output.strip() else '(Success, no output)'
        except subprocess.TimeoutExpired:
            return f'Error: Command timed out after {self.timeout} seconds.'
        except Exception as e:
            return f'Error: {e}'

    def commit_to_host(self, container_id: str, target_dir: str):
        """Copy container /workspace to host."""
        logger.info(f'💾 Copying files from container {container_id[:12]} to {target_dir}')
        try:
            subprocess.run(
                ['docker', 'cp', f'{container_id}:/workspace/.', target_dir],
                capture_output=True,
                timeout=120
            )
            logger.info('✅ Files copied successfully.')
        except Exception as e:
            logger.error(f'Failed to copy files from container: {e}')
            raise

    def cleanup_container(self, container_id: str):
        """Remove container."""
        try:
            subprocess.run(['docker', 'rm', '-f', container_id], capture_output=True, timeout=10)
            logger.debug(f'🗑️ Container {container_id[:12]} removed.')
        except Exception as e:
            logger.warning(f'Failed to remove container: {e}')

    def cleanup_all(self):
        """Remove all managed containers."""
        for branch_id, container_id in list(self._containers.items()):
            self.cleanup_container(container_id)
        self._containers.clear()

    def get_container(self, iteration: int, branch: int) -> str:
        """Get container ID for a branch."""
        branch_id = f'{iteration}_{branch}'
        return self._containers.get(branch_id)


_sandbox_instance = None


def get_sandbox(config):
    """Get Docker sandbox instance."""
    global _sandbox_instance
    if _sandbox_instance is None:
        _sandbox_instance = DockerSandbox(config)
        logger.info('🐳 Using Docker sandbox mode')
    return _sandbox_instance


def get_docker_sandbox(config):
    """Deprecated: Use get_sandbox() instead."""
    return get_sandbox(config)

