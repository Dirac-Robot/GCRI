import os
import shutil
import subprocess
import uuid
from loguru import logger


class DockerSandbox:
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
                result = subprocess.run(['docker', 'version'], capture_output=True, timeout=5)
                self._docker_available = result.returncode == 0
            except Exception:
                self._docker_available = False
        return self._docker_available

    def setup_branch(self, iteration: int, branch: int, source_dir: str) -> str:
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
            # Create baseline marker for tracking file changes
            self._execute_in_container(container_id, ['touch', '/workspace/.gcri_baseline'])
            self._containers[branch_id] = container_id
            logger.debug(f'🐳 Docker container created: {container_id[:12]} for branch {branch_id}')
            return container_id
        except Exception as e:
            logger.error(f'Failed to setup Docker branch: {e}')
            raise

    def _copy_to_container(self, source_dir: str, container_id: str):
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
            self._execute_in_container(container_id, ['pip', 'install', '-q', '-e', '/workspace'])

    def execute_command(self, container_id: str, command: str) -> str:
        return self._execute_in_container(container_id, ['sh', '-c', command])

    def execute_python(self, container_id: str, code: str) -> str:
        script_name = f'_script_{uuid.uuid4().hex[:8]}.py'
        write_cmd = f"cat > /workspace/{script_name} << 'GCRI_EOF'\n{code}\nGCRI_EOF"
        self._execute_in_container(container_id, ['sh', '-c', write_cmd])
        result = self._execute_in_container(container_id, ['python', script_name])
        self._execute_in_container(container_id, ['rm', '-f', script_name])
        return result

    def _execute_in_container(self, container_id: str, command: list, stdout_only: bool = False) -> str:
        exec_cmd = ['docker', 'exec', container_id]+command
        try:
            result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout+5
            )
            if stdout_only:
                if result.returncode != 0:
                    return f'Error (exit {result.returncode}): {result.stderr.strip()}'
                return result.stdout
            output = result.stdout+result.stderr
            if result.returncode != 0 and not output.strip():
                output = f'Exit Code {result.returncode}'
            return output if output.strip() else '(Success, no output)'
        except subprocess.TimeoutExpired:
            return f'Error: Command timed out after {self.timeout} seconds.'
        except Exception as e:
            return f'Error: {e}'

    def commit_to_host(self, container_id: str, target_dir: str):
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

    def clean_up_container(self, container_id: str):
        try:
            subprocess.run(['docker', 'rm', '-f', container_id], capture_output=True, timeout=10)
            logger.debug(f'🗑️ Container {container_id[:12]} removed.')
        except Exception as e:
            logger.warning(f'Failed to remove container: {e}')

    def merge_containers(self, source_containers: list, source_dir: str) -> str:
        """
        Create a new container with merged files from multiple source containers.

        Files from later containers override files from earlier ones if paths conflict.

        Args:
            source_containers: List of container IDs to merge from.
            source_dir: Original project directory for base setup.

        Returns:
            New container ID with merged files.
        """
        if not source_containers:
            raise ValueError('No source containers provided for merge.')

        if len(source_containers) == 1:
            # Only one source, just return it (no merge needed)
            return source_containers[0]

        merge_id = uuid.uuid4().hex[:8]
        container_name = f'gcri_merged_{merge_id}'
        temp_merge_dir = f'/tmp/gcri_merge_{merge_id}'

        try:
            os.makedirs(temp_merge_dir, exist_ok=True)

            # Copy files from each container in order (later overwrites earlier)
            for i, container_id in enumerate(source_containers):
                logger.debug(f'🔀 Merging files from container {i+1}/{len(source_containers)}: {container_id[:12]}')
                container_temp = f'{temp_merge_dir}/source_{i}'
                os.makedirs(container_temp, exist_ok=True)

                # Copy from container to temp
                copy_result = subprocess.run(
                    ['docker', 'cp', f'{container_id}:/workspace/.', container_temp],
                    capture_output=True,
                    timeout=60
                )
                if copy_result.returncode != 0:
                    logger.warning(f'Failed to copy from container {container_id[:12]}: {copy_result.stderr}')
                    continue

                # Merge into main temp dir (overwrite existing)
                for item in os.listdir(container_temp):
                    src = os.path.join(container_temp, item)
                    dst = os.path.join(temp_merge_dir, 'merged', item)
                    os.makedirs(os.path.join(temp_merge_dir, 'merged'), exist_ok=True)

                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            # Merge directories recursively
                            self._merge_directories(src, dst)
                        else:
                            shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)

            # Create new container
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
            result = subprocess.run(create_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f'Failed to create merged container: {result.stderr}')

            new_container_id = result.stdout.strip()
            subprocess.run(['docker', 'start', new_container_id], capture_output=True, timeout=10)

            # Copy merged files to new container
            merged_dir = os.path.join(temp_merge_dir, 'merged')
            if os.path.exists(merged_dir):
                subprocess.run(
                    ['docker', 'cp', f'{merged_dir}/.', f'{new_container_id}:/workspace'],
                    capture_output=True,
                    timeout=60
                )

            logger.info(f'🔀 Created merged container {new_container_id[:12]} from {len(source_containers)} sources')
            return new_container_id

        except Exception as e:
            logger.error(f'Failed to merge containers: {e}')
            raise
        finally:
            if os.path.exists(temp_merge_dir):
                shutil.rmtree(temp_merge_dir)

    def _merge_directories(self, src: str, dst: str):
        """Recursively merge src directory into dst, overwriting conflicts."""
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    self._merge_directories(s, d)
                else:
                    shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

    def clean_up_all(self):
        for branch_id, container_id in list(self._containers.items()):
            self.clean_up_container(container_id)
        self._containers.clear()

    def get_container(self, iteration: int, branch: int) -> str:
        branch_id = f'{iteration}_{branch}'
        return self._containers.get(branch_id)

    def clone_container(self, source_container_id: str, iteration: int, branch: int) -> str:
        """
        Create a new branch container by cloning an existing container's filesystem.

        Args:
            source_container_id: Container ID to clone from.
            iteration: Iteration index for the new container.
            branch: Branch index for the new container.

        Returns:
            New container ID with cloned files.
        """
        if not self.docker_available:
            raise RuntimeError('Docker is not available on this system.')

        branch_id = f'{iteration}_{branch}'
        container_name = f'gcri_branch_{branch_id}_{uuid.uuid4().hex[:8]}'
        temp_dir = f'/tmp/gcri_clone_{uuid.uuid4().hex[:8]}'

        try:
            os.makedirs(temp_dir, exist_ok=True)

            # Copy files from source container
            subprocess.run(
                ['docker', 'cp', f'{source_container_id}:/workspace/.', temp_dir],
                capture_output=True,
                timeout=60
            )

            # Create new container
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
            result = subprocess.run(create_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f'Failed to create container: {result.stderr}')

            container_id = result.stdout.strip()
            subprocess.run(['docker', 'start', container_id], capture_output=True, timeout=10)

            # Copy cloned files
            subprocess.run(
                ['docker', 'cp', f'{temp_dir}/.', f'{container_id}:/workspace'],
                capture_output=True,
                timeout=60
            )

            # Create baseline marker
            self._execute_in_container(container_id, ['touch', '/workspace/.gcri_baseline'])

            self._containers[branch_id] = container_id
            logger.debug(f'🔄 Cloned container {source_container_id[:12]} -> {container_id[:12]} for branch {branch_id}')
            return container_id

        except Exception as e:
            logger.error(f'Failed to clone container: {e}')
            raise
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def get_container_files(self, container_id: str) -> dict:
        """
        Get files that were created/modified by LLM in the container.

        Uses .gcri_baseline marker file to identify only files newer than baseline.

        Args:
            container_id: Container ID to inspect.

        Returns:
            Dict mapping file paths to their contents.
        """
        try:
            # Find files newer than baseline
            result = self._execute_in_container(
                container_id,
                ['find', '/workspace', '-newer', '/workspace/.gcri_baseline',
                 '-type', 'f', '!', '-name', '.gcri_baseline', '!', '-path', '*/__pycache__/*'],
                stdout_only=True
            )
            if result.startswith('Error'):
                logger.warning(f'Failed to find modified files: {result}')
                return {}

            files = {}
            for line in result.strip().split('\n'):
                line = line.strip()
                if not line or line == '(Success, no output)':
                    continue
                # Get file content
                content = self._execute_in_container(container_id, ['cat', line], stdout_only=True)
                if not content.startswith('Error (exit'):
                    # Convert to relative path
                    rel_path = line.replace('/workspace/', '', 1)
                    files[rel_path] = content
            return files

        except Exception as e:
            logger.warning(f'Failed to get container files: {e}')
            return {}


_sandbox_instance = None


def get_sandbox(config):
    global _sandbox_instance
    if _sandbox_instance is None:
        _sandbox_instance = DockerSandbox(config)
        logger.info('🐳 Using Docker sandbox mode')
    return _sandbox_instance


def get_docker_sandbox(config):
    return get_sandbox(config)
