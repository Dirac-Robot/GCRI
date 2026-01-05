import os
import shutil
import subprocess
import uuid
from loguru import logger


class LocalSandbox:
    """
    Executes code directly on the host filesystem without Docker.

    Uses temporary directories for branch isolation.
    Suitable for environments where Docker is unavailable or causes docker-in-docker issues.
    """

    def __init__(self, config):
        self.timeout = config.sandbox.timeout
        self._branches = {}

    def setup_branch(self, iteration: int, branch: int, source_dir: str) -> str:
        """
        Create isolated directory for a branch and copy project files.

        Args:
            iteration: Current iteration index
            branch: Branch index
            source_dir: Project directory to copy

        Returns:
            branch_id: Unique identifier for this branch (directory path)
        """
        branch_id = f'{iteration}_{branch}'
        branch_dir = f'/tmp/gcri_local_{branch_id}_{uuid.uuid4().hex[:8]}'
        ignore_patterns = {'.git', '__pycache__', 'venv', 'env', 'node_modules', '.idea', '.vscode', '.gcri'}
        try:
            os.makedirs(branch_dir, exist_ok=True)
            for item in os.listdir(source_dir):
                if item in ignore_patterns:
                    continue
                src = os.path.join(source_dir, item)
                dst = os.path.join(branch_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*ignore_patterns))
                else:
                    shutil.copy2(src, dst)
            self._branches[branch_id] = branch_dir
            logger.debug(f'📁 Local branch created: {branch_dir} for branch {branch_id}')
            return branch_dir
        except Exception as e:
            logger.error(f'Failed to setup local branch: {e}')
            raise

    def execute_command(self, branch_dir: str, command: str) -> str:
        """Execute shell command in branch directory."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=branch_dir,
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

    def execute_python(self, branch_dir: str, code: str) -> str:
        """Execute Python code in branch directory."""
        script_name = f'_script_{uuid.uuid4().hex[:8]}.py'
        script_path = os.path.join(branch_dir, script_name)
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)
            result = self.execute_command(branch_dir, f'python {script_name}')
            return result
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

    def _execute_in_container(self, branch_dir: str, command: list) -> str:
        """
        Execute command in branch directory.
        This method provides compatibility with DockerSandbox interface.
        """
        if command[0] == 'cat' and len(command) == 2:
            file_path = command[1]
            if not file_path.startswith('/'):
                file_path = os.path.join(branch_dir, file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                return f'Error: File not found: {file_path}'
            except Exception as e:
                return f'Error: {e}'
        elif command[0] == 'mkdir' and '-p' in command:
            dir_path = command[-1]
            if not dir_path.startswith('/'):
                dir_path = os.path.join(branch_dir, dir_path)
            os.makedirs(dir_path, exist_ok=True)
            return '(Success, no output)'
        elif command[0] == 'sh' and command[1] == '-c':
            return self.execute_command(branch_dir, command[2])
        elif command[0] == 'ls':
            return self.execute_command(branch_dir, f'ls {" ".join(command[1:])}')
        elif command[0] == 'rm':
            return self.execute_command(branch_dir, ' '.join(command))
        elif command[0] == 'python':
            return self.execute_command(branch_dir, f'python {command[1]}')
        else:
            return self.execute_command(branch_dir, ' '.join(command))

    @classmethod
    def commit_to_host(cls, branch_dir: str, target_dir: str):
        """Copy branch directory to host project."""
        logger.info(f'💾 Copying files from {branch_dir} to {target_dir}')
        try:
            for item in os.listdir(branch_dir):
                src = os.path.join(branch_dir, item)
                dst = os.path.join(target_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            logger.info('✅ Files copied successfully.')
        except Exception as e:
            logger.error(f'Failed to copy files: {e}')
            raise

    @classmethod
    def clean_up_container(cls, branch_dir: str):
        """Remove branch directory."""
        try:
            if os.path.exists(branch_dir):
                shutil.rmtree(branch_dir)
            logger.debug(f'🗑️ Branch directory {branch_dir} removed.')
        except Exception as e:
            logger.warning(f'Failed to remove branch directory: {e}')

    def clean_up_all(self):
        """Remove all managed branch directories."""
        for branch_id, branch_dir in list(self._branches.items()):
            self.clean_up_container(branch_dir)
        self._branches.clear()

    def get_container(self, iteration: int, branch: int) -> str:
        """Get branch directory path for a branch."""
        branch_id = f'{iteration}_{branch}'
        return self._branches.get(branch_id)
