import pytest
from gcri.tools.cli import SecurityPolicy


@pytest.fixture
def basic_policy():
    white_lists = {
        'sensitive_commands': ['rm', 'shutdown'],
        'sensitive_paths': ['/etc', '/var'],
        'sensitive_modules': {'os', 'sys'},
        'safe_extensions': {'.py', '.txt'}
    }
    return SecurityPolicy(white_lists)


def test_sensitive_command_detection(basic_policy):
    is_sensitive, reason = basic_policy.is_sensitive(
        'execute_shell_command',
        {'command': 'rm -rf /'}
    )
    assert is_sensitive is True
    assert 'rm' in reason

    is_sensitive, reason = basic_policy.is_sensitive(
        'execute_shell_command',
        {'command': 'ls -la'}
    )
    assert is_sensitive is False
    assert reason is None


def test_file_path_security(basic_policy):
    is_sensitive, reason = basic_policy.is_sensitive(
        'write_file',
        {'filepath': '/etc/passwd'}
    )
    assert is_sensitive is True
    assert '/etc' in reason

    is_sensitive, reason = basic_policy.is_sensitive(
        'write_file',
        {'filepath': 'test.py'}
    )
    assert is_sensitive is False


def test_python_code_analysis(basic_policy):
    code = "import os\nos.system('ls')"
    is_sensitive, reason = basic_policy.is_sensitive(
        'local_python_interpreter',
        {'code': code}
    )
    assert is_sensitive is True
    assert 'os' in reason

    code = "print('hello')"
    is_sensitive, reason = basic_policy.is_sensitive(
        'local_python_interpreter',
        {'code': code}
    )
    assert is_sensitive is False
