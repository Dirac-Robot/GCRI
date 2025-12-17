import json
import os
import shutil
from datetime import datetime
from loguru import logger


class SandboxManager:
    def __init__(self, config):
        self.config = config
        self._project_dir = config.project_dir
        self._run_dir = config.run_dir
        self._work_dir = None
        self._log_dir = None
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

    def setup(self):
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._work_dir = os.path.join(self.run_dir, f'run-{timestamp}')
        self._log_dir = os.path.join(self.work_dir, 'logs')
        logger.info(f'📦 Creating workspaces in sandbox at: {self.work_dir}')
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _smart_copy(self, src, dst, *, follow_symlinks=True):
        limit_bytes = self.config.protocols.max_copy_size*1024*1024
        try:
            if os.path.islink(src):
                link_to = os.readlink(src)
                os.symlink(link_to, dst)
            elif os.path.getsize(src) > limit_bytes:
                os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copy2(src, dst)
        except Exception as e:
            logger.warning(f'Smart copy failed for {src}: {e}')

    def setup_branch(self, iteration_count, branch_index):
        root_dir = os.path.join(self.work_dir, 'workspaces')
        os.makedirs(root_dir, exist_ok=True)
        branch_workspace = os.path.join(root_dir, f'iter_{iteration_count}_branch_{branch_index}')

        ignore = shutil.ignore_patterns(
            '.git',
            '__pycache__',
            'venv',
            'env',
            'node_modules',
            '.idea',
            '.vscode',
            '.gcri',
            '*.pyc'
        )

        if os.path.exists(branch_workspace):
            shutil.rmtree(branch_workspace)
        os.makedirs(branch_workspace, exist_ok=True)

        shutil.copytree(
            self.project_dir,
            branch_workspace,
            ignore=ignore,
            copy_function=self._smart_copy,
            dirs_exist_ok=True
        )
        return branch_workspace

    def get_branch_context(self, iteration_count, num_results):
        file_contexts = []
        workspace_root = os.path.join(self.work_dir, 'workspaces')
        for i in range(num_results):
            branch_dir = os.path.join(workspace_root, f'iter_{iteration_count}_branch_{i}')
            if os.path.exists(branch_dir):
                rel_path = os.path.relpath(branch_dir, start=self.work_dir)
                file_contexts.append(f'- Branch {i} files location: {rel_path}')
            else:
                file_contexts.append(f'- Branch {i} files location: (Directory not found)')
        return '\n'.join(file_contexts)

    def save_iteration_log(self, index, result_data):
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f'log_iteration_{index:02d}.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        logger.info(f'Result of iteration {index} saved to: {log_path}')
        return log_path

    def get_winning_branch_path(self, index, branch_index):
        return os.path.join(
            self.work_dir, 'workspaces', f'iter_{index}_branch_{branch_index}'
        )

    def commit_winning_branch(self, winning_branch_path):
        logger.info(f'💾 Merging changes from winning branch: {winning_branch_path}')
        ignore_patterns = {'.git', '__pycache__', 'venv', 'env', '.idea', 'workspaces'}
        for root, dirs, files in os.walk(winning_branch_path):
            dirs[:] = [d for d in dirs if d not in ignore_patterns]
            rel_path = os.path.relpath(root, winning_branch_path)
            target_dir = os.path.join(self.project_dir, rel_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            for file in files:
                if file in ignore_patterns or file.endswith('.pyc'):
                    continue
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, file)
                if os.path.islink(src_file):
                    continue
                try:
                    shutil.copy2(src_file, dst_file)
                except Exception as e:
                    logger.error(f'Failed to copy {src_file}: {e}')
        logger.info('✅ Merge completed successfully.')
