class MockSandboxManager:
    def __init__(self, config=None):
        self.config = config
        self.fs = {}
        self.work_dir = '/mock/workspace'
        self.command_history = []
        self.setup_branch_calls = []
        self._branch_containers = {}

    def setup(self):
        pass

    def setup_branch(self, iteration: int, branch: int) -> str:
        container_id = f'mock_container_{iteration}_{branch}'
        self._branch_containers[(iteration, branch)] = container_id
        self.setup_branch_calls.append((iteration, branch))
        return container_id

    def execute_command(self, container_id, command):
        self.command_history.append((container_id, command))
        if command.startswith('mkdir -p'):
            pass
        return 'Command Executed'

    def execute_python(self, container_id, code):
        self.command_history.append((container_id, f'python: {code}'))
        return 'Python Executed'

    def read_file(self, path):
        return self.fs.get(path, '')

    def write_file(self, path, content):
        self.fs[path] = content

    def get_winning_branch_path(self, iteration, branch_index):
        return f'{self.work_dir}/iter_{iteration}/branch_{branch_index}'

    def commit_winning_branch(self, branch_path):
        self.command_history.append(f'COMMIT: {branch_path}')

    def save_iteration_log(self, iteration, result):
        pass

    def get_branch_context(self, iteration, num_branches):
        return 'Mock File Context'
