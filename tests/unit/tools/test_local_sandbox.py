import os
import shutil
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from gcri.tools.local_sandbox import LocalSandbox


class MockConfig:
    def __init__(self):
        self.sandbox = MagicMock()
        self.sandbox.timeout = 30
        self.sandbox.get = lambda key, default=None: getattr(self.sandbox, key, default)


class TestLocalSandbox:
    @pytest.fixture
    def config(self):
        return MockConfig()

    @pytest.fixture
    def sandbox(self, config):
        return LocalSandbox(config)

    @pytest.fixture
    def temp_source_dir(self):
        temp_dir = tempfile.mkdtemp()
        with open(os.path.join(temp_dir, 'test.py'), 'w') as f:
            f.write('print("hello")')
        with open(os.path.join(temp_dir, 'readme.txt'), 'w') as f:
            f.write('Test readme')
        os.makedirs(os.path.join(temp_dir, 'subdir'))
        with open(os.path.join(temp_dir, 'subdir', 'nested.py'), 'w') as f:
            f.write('x = 1')
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_setup_branch_creates_directory(self, sandbox, temp_source_dir):
        """Test that setup_branch creates a branch directory and copies files."""
        branch_dir = sandbox.setup_branch(0, 0, temp_source_dir)
        assert os.path.isdir(branch_dir)
        assert os.path.exists(os.path.join(branch_dir, 'test.py'))
        assert os.path.exists(os.path.join(branch_dir, 'readme.txt'))
        assert os.path.exists(os.path.join(branch_dir, 'subdir', 'nested.py'))
        sandbox.cleanup_container(branch_dir)

    def test_setup_branch_ignores_patterns(self, sandbox, temp_source_dir):
        """Test that setup_branch ignores .git, __pycache__, etc."""
        os.makedirs(os.path.join(temp_source_dir, '.git'))
        os.makedirs(os.path.join(temp_source_dir, '__pycache__'))
        with open(os.path.join(temp_source_dir, '.git', 'config'), 'w') as f:
            f.write('git config')
        branch_dir = sandbox.setup_branch(0, 1, temp_source_dir)
        assert not os.path.exists(os.path.join(branch_dir, '.git'))
        assert not os.path.exists(os.path.join(branch_dir, '__pycache__'))
        sandbox.cleanup_container(branch_dir)

    def test_execute_command(self, sandbox, temp_source_dir):
        """Test execute_command runs shell commands."""
        branch_dir = sandbox.setup_branch(0, 0, temp_source_dir)
        result = sandbox.execute_command(branch_dir, 'echo "hello world"')
        assert 'hello world' in result
        sandbox.cleanup_container(branch_dir)

    def test_execute_python(self, sandbox, temp_source_dir):
        """Test execute_python runs Python code."""
        branch_dir = sandbox.setup_branch(0, 0, temp_source_dir)
        result = sandbox.execute_python(branch_dir, 'print(1+1)')
        assert '2' in result
        sandbox.cleanup_container(branch_dir)

    def test_execute_in_container_cat(self, sandbox, temp_source_dir):
        """Test _execute_in_container with cat command."""
        branch_dir = sandbox.setup_branch(0, 0, temp_source_dir)
        result = sandbox._execute_in_container(branch_dir, ['cat', 'test.py'])
        assert 'print("hello")' in result
        sandbox.cleanup_container(branch_dir)

    def test_execute_in_container_mkdir(self, sandbox, temp_source_dir):
        """Test _execute_in_container with mkdir command."""
        branch_dir = sandbox.setup_branch(0, 0, temp_source_dir)
        sandbox._execute_in_container(branch_dir, ['mkdir', '-p', 'newdir/deep'])
        assert os.path.isdir(os.path.join(branch_dir, 'newdir', 'deep'))
        sandbox.cleanup_container(branch_dir)

    def test_commit_to_host(self, sandbox, temp_source_dir):
        """Test commit_to_host copies files back."""
        branch_dir = sandbox.setup_branch(0, 0, temp_source_dir)
        new_file = os.path.join(branch_dir, 'new_file.txt')
        with open(new_file, 'w') as f:
            f.write('new content')
        target_dir = tempfile.mkdtemp()
        try:
            sandbox.commit_to_host(branch_dir, target_dir)
            assert os.path.exists(os.path.join(target_dir, 'new_file.txt'))
            with open(os.path.join(target_dir, 'new_file.txt')) as f:
                assert f.read() == 'new content'
        finally:
            shutil.rmtree(target_dir, ignore_errors=True)
            sandbox.cleanup_container(branch_dir)

    def test_cleanup_container(self, sandbox, temp_source_dir):
        """Test cleanup_container removes the branch directory."""
        branch_dir = sandbox.setup_branch(0, 0, temp_source_dir)
        assert os.path.exists(branch_dir)
        sandbox.cleanup_container(branch_dir)
        assert not os.path.exists(branch_dir)

    def test_cleanup_all(self, sandbox, temp_source_dir):
        """Test cleanup_all removes all branch directories."""
        branch_dir1 = sandbox.setup_branch(0, 0, temp_source_dir)
        branch_dir2 = sandbox.setup_branch(0, 1, temp_source_dir)
        assert os.path.exists(branch_dir1)
        assert os.path.exists(branch_dir2)
        sandbox.cleanup_all()
        assert not os.path.exists(branch_dir1)
        assert not os.path.exists(branch_dir2)

    def test_get_container(self, sandbox, temp_source_dir):
        """Test get_container returns correct branch directory."""
        branch_dir = sandbox.setup_branch(1, 2, temp_source_dir)
        retrieved = sandbox.get_container(1, 2)
        assert retrieved == branch_dir
        sandbox.cleanup_all()

    def test_command_timeout(self, config):
        """Test that commands time out."""
        config.sandbox.timeout = 1
        sandbox = LocalSandbox(config)
        temp_dir = tempfile.mkdtemp()
        try:
            result = sandbox.execute_command(temp_dir, 'sleep 10')
            assert 'timed out' in result.lower() or 'error' in result.lower()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
