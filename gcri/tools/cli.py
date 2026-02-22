import ast
import os
import sys
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any, List, Type

from ato.adict import ADict
from ddgs import DDGS
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input
from langchain_core.tools import tool
from loguru import logger
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from gcri.config import scope
from gcri.tools.docker_sandbox import get_sandbox

console = Console(force_terminal=True)


class GlobalVariables:
    CONTAINER_VAR = None
    AUTO_MODE_FILE = None
    CONFIG = None
    SANDBOX = None


@scope
def set_global_variables(config):
    GlobalVariables.CONTAINER_VAR = ContextVar('container_id', default=None)
    GlobalVariables.AUTO_MODE_FILE = os.path.join(config.project_dir, '.gcri_auto_mode')
    GlobalVariables.CONFIG = config


def get_cached_sandbox():
    """Get cached sandbox instance for cli tools."""
    if GlobalVariables.SANDBOX is None:
        GlobalVariables.SANDBOX = get_sandbox(GlobalVariables.CONFIG or scope.config)
    return GlobalVariables.SANDBOX





def get_input(message):
    logger.info(message)
    return sys.stdin.buffer.readline().decode('utf-8', errors='ignore').strip()


def get_container_id():
    return GlobalVariables.CONTAINER_VAR.get()


def _to_container_path(file_path: str) -> str:
    """Translate host absolute paths to container /workspace/ paths."""
    config = GlobalVariables.CONFIG
    if config and hasattr(config, 'project_dir') and config.project_dir:
        project_dir = config.project_dir.rstrip('/')
        if file_path.startswith(project_dir+'/'):
            return '/workspace'+file_path[len(project_dir):]
        if file_path == project_dir:
            return '/workspace'
    return file_path


@scope
def _get_black_and_white_lists(config):
    black_and_white_lists = ADict.from_file(config.templates.black_and_white_lists).to_dict()
    black_and_white_lists['safe_extensions'] = set(black_and_white_lists.get('safe_extensions', []))
    black_and_white_lists['sensitive_modules'] = set(black_and_white_lists.get('sensitive_modules', []))
    return black_and_white_lists


@tool
def execute_shell_command(command: str) -> str:
    """Executes a shell command in the sandbox."""
    container_id = get_container_id()
    if not container_id:
        return 'Error: No sandbox container available. Run within GCRI context.'
    config = GlobalVariables.CONFIG
    if config and hasattr(config, 'project_dir') and config.project_dir:
        command = command.replace(config.project_dir.rstrip('/'), '/workspace')
    sandbox = get_cached_sandbox()
    return sandbox.execute_command(container_id, command)


@tool
def read_file(file_path: str) -> str:
    """Reads the content of a file from the sandbox."""
    container_id = get_container_id()
    if not container_id:
        return f'Error: No sandbox container available.'
    sandbox = get_cached_sandbox()
    result = sandbox._execute_in_container(container_id, ['cat', _to_container_path(file_path)])
    return result


@tool
def write_file(file_path: str, content: str) -> str:
    """Writes content to a file in the sandbox."""
    container_id = get_container_id()
    if not container_id:
        return f'Error: No sandbox container available.'
    sandbox = get_cached_sandbox()
    container_path = _to_container_path(file_path)
    dir_path = os.path.dirname(container_path)
    if dir_path:
        sandbox._execute_in_container(container_id, ['mkdir', '-p', dir_path])
    write_cmd = f"cat > '{container_path}' << 'GCRI_WRITE_EOF'\n{content}\nGCRI_WRITE_EOF"
    result = sandbox._execute_in_container(container_id, ['sh', '-c', write_cmd])
    if 'Error' in result:
        return result
    return f'Successfully wrote to "{file_path}" in workspace.'


@tool
def list_directory(path: str = '.') -> str:
    """Lists files and directories at the given path in the sandbox.
    Returns type, size, and name for each entry.

    Args:
        path: Directory path to list. Defaults to project root.
    """
    container_id = get_container_id()
    if not container_id:
        return 'Error: No sandbox container available.'
    sandbox = get_cached_sandbox()
    container_path = _to_container_path(path)
    result = sandbox._execute_in_container(
        container_id, ['ls', '-la', '--group-directories-first', container_path]
    )
    return result


@tool
def search_files(pattern: str, path: str = '.') -> str:
    """Searches for files matching a glob pattern in the sandbox.
    Returns matching file paths (max 50 results).

    Args:
        pattern: Glob pattern to match (e.g. '*.py', 'test_*').
        path: Directory to search in. Defaults to project root.
    """
    container_id = get_container_id()
    if not container_id:
        return 'Error: No sandbox container available.'
    sandbox = get_cached_sandbox()
    container_path = _to_container_path(path)
    cmd = f"find {container_path} -name '{pattern}' -type f 2>/dev/null | head -50"
    result = sandbox._execute_in_container(container_id, ['sh', '-c', cmd])
    if not result.strip():
        return f'No files matching "{pattern}" found in {path}.'
    return result


@tool
def grep_in_files(pattern: str, path: str = '.', include: str = '*') -> str:
    """Searches for a text pattern inside files in the sandbox.
    Returns matching lines with file path and line number (max 50 results).

    Args:
        pattern: Text or regex pattern to search for.
        path: Directory to search in. Defaults to project root.
        include: File glob filter (e.g. '*.py'). Defaults to all files.
    """
    container_id = get_container_id()
    if not container_id:
        return 'Error: No sandbox container available.'
    sandbox = get_cached_sandbox()
    container_path = _to_container_path(path)
    cmd = f"grep -rn --include='{include}' '{pattern}' {container_path} 2>/dev/null | head -50"
    result = sandbox._execute_in_container(container_id, ['sh', '-c', cmd])
    if not result.strip():
        return f'No matches for "{pattern}" found in {path}.'
    return result


@tool
def local_python_interpreter(code: str) -> str:
    """Executes Python code in the sandbox."""
    container_id = get_container_id()
    if not container_id:
        return 'Error: No sandbox container available. Run within GCRI context.'
    sandbox = get_cached_sandbox()
    return sandbox.execute_python(container_id, code)


@tool
def search_web(query: str) -> str:
    """Searches the web using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return 'No results found.'
            formatted = []
            for result in results:
                formatted.append(f'Title: {result["title"]}\nLink: {result["href"]}\nSnippet: {result["body"]}')
            return '\n---\n'.join(formatted)
    except Exception as e:
        return f'Search Error: {e}'


class BranchContainerRegistry:
    """Registry for branch container IDs."""
    _containers = {}
    _current_iteration = 0

    @classmethod
    def set_containers(cls, iteration: int, containers: dict):
        cls._current_iteration = iteration
        cls._containers = containers

    @classmethod
    def get_container(cls, branch_index: int) -> str:
        return cls._containers.get((cls._current_iteration, branch_index))

    @classmethod
    def list_branches(cls) -> list:
        return [k[1] for k in cls._containers.keys() if k[0] == cls._current_iteration]


@tool
def read_branch_file(branch_index: int, file_path: str) -> str:
    """
    Reads a file from a specific branch's sandbox.

    Args:
        branch_index: The branch index (0-based)
        file_path: Path to the file within the sandbox's /workspace
    """
    container_id = BranchContainerRegistry.get_container(branch_index)
    if not container_id:
        return f'Error: Branch {branch_index} container not found.'
    sandbox = get_cached_sandbox()
    return sandbox._execute_in_container(container_id, ['cat', file_path])


@tool
def list_branch_files(branch_index: int, directory: str = '.') -> str:
    """
    Lists files in a directory within a specific branch's sandbox.

    Args:
        branch_index: The branch index (0-based)
        directory: Directory path to list (default: workspace root)
    """
    container_id = BranchContainerRegistry.get_container(branch_index)
    if not container_id:
        return f'Error: Branch {branch_index} container not found.'
    sandbox = get_cached_sandbox()
    return sandbox._execute_in_container(container_id, ['ls', '-la', directory])


@tool
def run_branch_command(branch_index: int, command: str) -> str:
    """
    Executes a shell command in a specific branch's sandbox.

    Args:
        branch_index: The branch index (0-based)
        command: Shell command to execute
    """
    container_id = BranchContainerRegistry.get_container(branch_index)
    if not container_id:
        return f'Error: Branch {branch_index} container not found.'
    sandbox = get_cached_sandbox()
    return sandbox.execute_command(container_id, command)


CLI_TOOLS = [execute_shell_command, read_file, write_file, list_directory, search_files, grep_in_files, local_python_interpreter]
DECISION_TOOLS = [read_branch_file, list_branch_files, run_branch_command]


class SecurityPolicy:
    """Pure security policy checker without UI dependencies."""

    def __init__(self, black_and_white_lists: dict):
        self._lists = black_and_white_lists

    def analyze_python_code(self, code: str) -> tuple[bool, str | None]:
        """Check Python code for sensitive operations."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [n.name.split('.')[0] for n in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        names = [node.module.split('.')[0]]
                    for name in names:
                        if name in self._lists.get('sensitive_modules', set()):
                            return True, f'Importing sensitive module "{name}"'
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == 'open':
                        return True, 'Using file "open()" function'
        except SyntaxError:
            return True, 'Syntax Error'
        except Exception as e:
            return True, f'Complex code detected: {e}'
        return False, None

    def is_sensitive(self, tool_name: str, args: dict) -> tuple[bool, str | None]:
        """Check if a tool invocation is sensitive."""
        if tool_name == 'execute_shell_command':
            command = args.get('command', '').strip()
            for sensitive_command in self._lists.get('sensitive_commands', []):
                if sensitive_command in command.split():
                    return True, f'Command "{sensitive_command}" detected'
        if tool_name == 'write_file':
            path_in_args = args.get('filepath', '')
            path = Path(path_in_args)
            for sensitive_path in self._lists.get('sensitive_paths', []):
                if sensitive_path in path_in_args:
                    return True, f'Sensitive path "{sensitive_path}" detected'
            if path.suffix not in self._lists.get('safe_extensions', set()):
                return True, f'Unusual extension "{path.suffix}"'
        if tool_name == 'local_python_interpreter':
            return self.analyze_python_code(args.get('code', ''))
        return False, None


class InteractiveToolGuard:
    """CLI-specific tool execution guard with user interaction."""
    _io_lock = threading.Lock()

    def __init__(self):
        self.tools = {t.name: t for t in CLI_TOOLS}
        try:
            black_and_white_lists = _get_black_and_white_lists()
        except Exception as e:
            logger.error(f'Cannot load black and white lists: {e}')
            black_and_white_lists = {
                'sensitive_commands': [],
                'sensitive_paths': [],
                'sensitive_modules': set(),
                'safe_extensions': set()
            }
        self._policy = SecurityPolicy(black_and_white_lists)

    @property
    def auto_mode(self):
        return os.path.exists(GlobalVariables.AUTO_MODE_FILE)

    @auto_mode.setter
    def auto_mode(self, value):
        if value:
            with open(GlobalVariables.AUTO_MODE_FILE, 'a'):
                os.utime(GlobalVariables.AUTO_MODE_FILE, None)
        else:
            if os.path.exists(GlobalVariables.AUTO_MODE_FILE):
                try:
                    os.remove(GlobalVariables.AUTO_MODE_FILE)
                except OSError:
                    pass


    def invoke(self, name, args, task_id):
        with self._io_lock:
            container_id = GlobalVariables.CONTAINER_VAR.get()
            container_display = container_id[:12] if container_id else 'None'
            console.print(
                Panel(f'Agent Request: [bold cyan]{name}[/]\nContainer: [dim]{container_display}[/]', border_style='blue')
            )
            if 'code' in args:
                console.print(Syntax(args['code'], 'python', theme='monokai', line_numbers=True))
            elif 'command' in args:
                console.print(Syntax(args['command'], 'bash', theme='monokai'))
            else:
                console.print(str(args))
            is_sensitive, reason = self._policy.is_sensitive(name, args)
            # Docker provides isolation, so we can auto-execute more freely
            is_safe_sandbox_op = True
            if is_safe_sandbox_op or (self.auto_mode and not is_sensitive):
                console.print(f'[bold green]⚡ Auto-Executing in Docker (Task {task_id})[/]')
                try:
                    result = self.tools[name].invoke(args)
                    console.print(f'[dim]Result: {str(result)[:100]}...[/]')
                    return str(result)
                except Exception as e:
                    return f'Tool Error: {e}'
            while True:
                console.print('[bold yellow]>> (y)es / (a)lways / (n)o / (e)dit : [/]', end='')
                sys.stdout.flush()
                choice = input().strip().lower()
                if choice == 'y':
                    console.print(f'[dim]Executing Task {task_id}...[/]')
                    try:
                        result = self.tools[name].invoke(args)
                        console.print('[bold green]Done.[/]')
                        return str(result)
                    except Exception as e:
                        return f'Tool Error: {e}'
                elif choice == 'a':
                    console.print('[bold green]⚡ Auto-Mode Enabled.[/]')
                    self.auto_mode = True
                    try:
                        result = self.tools[name].invoke(args)
                        console.print('[bold green]Done.[/]')
                        return str(result)
                    except Exception as e:
                        return f'Tool Error: {e}'
                elif choice == 'n':
                    console.print('[bold red]Denied.[/]')
                    return 'User denied execution.'
                elif choice == 'e':
                    console.print('[bold magenta]Manual Override (Type "EOF" to finish)[/]')
                    lines = []
                    while True:
                        line = input()
                        if line.strip() == 'EOF':
                            break
                        lines.append(line)
                    new_value = '\n'.join(lines)
                    if not new_value.strip():
                        console.print('No input provided.')
                        continue
                    if name == 'execute_shell_command':
                        args['command'] = new_value
                    elif name == 'write_file':
                        args['content'] = new_value
                    elif name == 'local_python_interpreter':
                        args['code'] = new_value
                    console.print('[dim]Executing modified...[/]')
                    try:
                        result = self.tools[name].invoke(args)
                        console.print('[bold green]Done.[/]')
                        return str(result)
                    except Exception as e:
                        return f'Tool Error: {e}'


SHARED_GUARD = None


def get_shared_guard():
    global SHARED_GUARD
    if SHARED_GUARD is None:
        SHARED_GUARD = InteractiveToolGuard()
    return SHARED_GUARD


class RecursiveToolAgent(Runnable):
    def __init__(self, agent, schema: Type[BaseModel], tools: List[Any], container_id: str = None, max_recursion_depth=50):
        self.agent = agent
        self.schema = schema
        self.tools_map = {t.name: t for t in tools}
        unique_tools = list({t.name: t for t in tools}.values())
        self.model_with_tools = self.agent.bind_tools(unique_tools+[schema])
        self.guard = get_shared_guard()
        self.container_id = container_id
        self.max_recursion_depth = max_recursion_depth

    def invoke(self, input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        token = None
        if self.container_id:
            token = GlobalVariables.CONTAINER_VAR.set(self.container_id)
        try:
            if isinstance(input, str):
                messages = [HumanMessage(content=input)]
            elif isinstance(input, dict):
                messages = [HumanMessage(content=str(input))]
            else:
                messages = list(input)
            recursion_count = 0
            parse_retry_count = 0
            max_parse_retries = 3
            while True:
                result = self.model_with_tools.invoke(messages, config=config)
                messages.append(result)
                if not result.tool_calls:
                    # Handle empty response (content can be str or list)
                    content = result.content
                    is_empty = False
                    if content is None:
                        is_empty = True
                    elif isinstance(content, str):
                        is_empty = not content.strip()
                    elif isinstance(content, list):
                        is_empty = len(content) == 0 or all(
                            (isinstance(c, str) and not c.strip()) or
                            (isinstance(c, dict) and not c.get('text', '').strip())
                            for c in content
                        )
                    if is_empty:
                        logger.warning(f'Empty response from model (content type: {type(content).__name__}). Retrying...')
                        messages.append(HumanMessage(content='Your response was empty. Please provide the final answer or call a tool.'))
                        recursion_count += 1
                        if self.max_recursion_depth is not None and recursion_count >= self.max_recursion_depth:
                            logger.error('Max recursion depth reached with empty responses.')
                            return None
                        continue
                    # Try structured output parsing with retry on failure
                    try:
                        return self.agent.with_structured_output(self.schema).invoke(messages)
                    except Exception as e:
                        parse_retry_count += 1
                        if parse_retry_count >= max_parse_retries:
                            logger.error(f'Structured output parsing failed after {max_parse_retries} retries: {e}')
                            return None
                        logger.warning(f'JSON parsing error (attempt {parse_retry_count}/{max_parse_retries}): {e}')
                        messages.append(HumanMessage(content='Your response had invalid JSON format. Please provide a valid JSON response matching the required schema.'))
                        continue
                outputs = []
                for call in result.tool_calls:
                    name, args, call_id = call['name'], call['args'], call['id']
                    if name == self.schema.__name__:
                        try:
                            return self.schema(**args)
                        except Exception as e:
                            logger.error(f'Unknown error is occurred during building schema for tool-calling: {e}')
                            return
                    if name in self.tools_map:
                        tool_fn = self.tools_map[name]
                        if name in self.guard.tools:
                            output = self.guard.invoke(name, args, call_id)
                        else:
                            try:
                                output = tool_fn.invoke(args)
                            except Exception as e:
                                output = f'Tool Error: {e}'
                        output_str = str(output)
                        outputs.append(ToolMessage(content=output_str, tool_call_id=call_id, name=name))
                messages.extend(outputs)
                recursion_count += 1
                if self.max_recursion_depth is not None and recursion_count >= self.max_recursion_depth:
                    break
            raise ValueError(
                f'Maximum recursion depth is {self.max_recursion_depth}, '
                f'and it is exceeded while tool calling.'
            )
        finally:
            if token:
                GlobalVariables.CONTAINER_VAR.reset(token)


class CodeAgentBuilder:
    def __init__(self, model_id, tools=None, container_id=None, max_recursion_depth=50, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.tools = tools or []
        self.container_id = container_id
        self.max_recursion_depth = max_recursion_depth
        self._agent = init_chat_model(self.model_id, **kwargs)

    @property
    def agent(self):
        return self._agent

    def with_structured_output(self, schema: Type[BaseModel]):
        return RecursiveToolAgent(
            self.agent,
            schema,
            tools=self.tools,
            container_id=self.container_id,
            max_recursion_depth=self.max_recursion_depth
        )

    def invoke(self, *args, **kwargs):
        return self.agent.invoke(*args, **kwargs)


def build_model(model_id, gcri_options=None, container_id=None, **parameters):
    if gcri_options is not None:
        use_code_tools = gcri_options.get('use_code_tools', False)
        use_web_search = gcri_options.get('use_web_search', False)
        max_recursion_depth = gcri_options.get('max_recursion_depth', 50)
    else:
        use_code_tools = False
        use_web_search = False
        max_recursion_depth = 50
    tools = CLI_TOOLS if use_code_tools else []
    tools += [search_web] if use_web_search else []
    return CodeAgentBuilder(
        model_id,
        tools=tools,
        container_id=container_id,
        max_recursion_depth=max_recursion_depth,
        **parameters
    )


def build_decision_model(model_id, gcri_options=None, **parameters):
    """Model builder for Decision agent. Uses DECISION_TOOLS when use_code_tools=True."""
    if gcri_options is not None:
        use_code_tools = gcri_options.get('use_code_tools', False)
        max_recursion_depth = gcri_options.get('max_recursion_depth', 50)
    else:
        use_code_tools = False
        max_recursion_depth = 50
    tools = DECISION_TOOLS if use_code_tools else []
    return CodeAgentBuilder(
        model_id,
        tools=tools,
        container_id=None,
        max_recursion_depth=max_recursion_depth,
        **parameters
    )
