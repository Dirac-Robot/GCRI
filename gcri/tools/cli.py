import ast
import os
import subprocess
import sys
import threading
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, List, Type

from duckduckgo_search import DDGS
from loguru import logger

from ato.adict import ADict
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input
from langchain_core.tools import tool
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from gcri.config import scope

console = Console(force_terminal=True)
CWD_VAR = ContextVar('cwd', default='.')
PROJECT_ROOT = os.getcwd()
AUTO_MODE_FILE = os.path.join(PROJECT_ROOT, '.gcri_auto_mode')
logger.info(f'AUTO MODE FILE SET TO: {AUTO_MODE_FILE}')


def get_cwd():
    d = CWD_VAR.get()
    os.makedirs(d, exist_ok=True)
    return d


def get_environment_with_python_path(cwd):
    environment = os.environ.copy()
    prev_python_path = environment.get('PYTHONPATH', '')
    next_python_path = f'{cwd}{os.pathsep}{PROJECT_ROOT}{os.pathsep}{prev_python_path}'
    environment['PYTHONPATH'] = next_python_path
    environment['PYTHONUNBUFFERED'] = '1'
    return environment


@scope
def _get_black_and_white_lists(config):
    black_and_white_lists = ADict.from_file(config.templates.black_and_white_lists).to_dict()
    black_and_white_lists['safe_extensions'] = set(black_and_white_lists.get('safe_extensions', []))
    black_and_white_lists['sensitive_modules'] = set(black_and_white_lists.get('sensitive_modules', []))
    return black_and_white_lists


@tool
def execute_shell_command(command: str) -> str:
    """Executes a shell command."""
    cwd = get_cwd()
    environment = get_environment_with_python_path(cwd)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=cwd,
            env=environment
        )
        if result.returncode == 0:
            return result.stdout if result.stdout.strip() else '(Success, no output)'
        return f'Exit Code {result.returncode}:\n{result.stderr}'
    except subprocess.TimeoutExpired:
        return 'Error: Command timed out.'
    except Exception as e:
        return f'Error: {e}'


@tool
def read_file(filepath: str) -> str:
    """Reads the content of a file."""
    cwd = get_cwd()
    target = os.path.join(cwd, filepath)
    if not os.path.exists(target):
        return f'Error: File "{filepath}" not found in {cwd}.'
    try:
        with open(target, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f'Error: {e}'


@tool
def write_file(filepath: str, content: str) -> str:
    """Writes content to a file."""
    cwd = get_cwd()
    target = os.path.join(cwd, filepath)
    try:
        os.makedirs(os.path.dirname(target) or '.', exist_ok=True)
        with open(target, 'w', encoding='utf-8') as f:
            f.write(content)
        return f'Successfully wrote to "{filepath}" in workspace.'
    except Exception as e:
        return f'Error: {e}'


@tool
def local_python_interpreter(code: str) -> str:
    """Executes Python code locally."""
    cwd = get_cwd()
    environment = get_environment_with_python_path(cwd)
    script_name = f'_script_{uuid.uuid4().hex[:8]}.py'
    target = os.path.join(cwd, script_name)
    try:
        with open(target, 'w', encoding='utf-8') as f:
            f.write(code)
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd,
            env=environment
        )
        try:
            os.remove(target)
        except:
            pass
        output = result.stdout+result.stderr
        if result.returncode != 0:
            return f'Execution failed (Exit {result.returncode}):\n{output}'
        return output if output.strip() else '(No output printed)'
    except Exception as e:
        return f'Error running python: {e}'


@tool
def search_web(query: str, max_results=5) -> str:
    """Searches the web using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return 'No results found.'
            formatted = []
            for r in results:
                formatted.append(f'Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}')
            return '\n---\n'.join(formatted)
    except Exception as e:
        return f'Search Error: {e}'


CLI_TOOLS = [execute_shell_command, read_file, write_file, local_python_interpreter]


class InteractiveToolGuard:
    _io_lock = threading.Lock()

    def __init__(self):
        self.tools = {t.name: t for t in CLI_TOOLS}
        try:
            self._black_and_white_lists = _get_black_and_white_lists()
        except:
            self._black_and_white_lists = {
                'sensitive_commands': [],
                'sensitive_paths': [],
                'sensitive_modules': set(),
                'safe_extensions': set()
            }

    @property
    def auto_mode(self):
        return os.path.exists(AUTO_MODE_FILE)

    @auto_mode.setter
    def auto_mode(self, value):
        if value:
            with open(AUTO_MODE_FILE, 'a'):
                os.utime(AUTO_MODE_FILE, None)
        else:
            if os.path.exists(AUTO_MODE_FILE):
                try:
                    os.remove(AUTO_MODE_FILE)
                except OSError:
                    pass

    def _analyze_python_code(self, code: str):
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
                        if name in self._black_and_white_lists['sensitive_modules']:
                            return True, f'Importing sensitive module "{name}"'
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == 'open':
                        return True, 'Using file "open()" function'
        except SyntaxError:
            return True, 'Syntax Error'
        except:
            return True, 'Complex code detected'
        return False, None

    def _is_sensitive(self, name, args):
        if name == 'execute_shell_command':
            command = args.get('command', '').strip()
            for sensitive_command in self._black_and_white_lists.get('sensitive_commands', []):
                if sensitive_command in command.split():
                    return True, f'Command "{sensitive_command}" detected'
        if name == 'write_file':
            path_in_args = args.get('filepath', '')
            path = Path(path_in_args)
            for sensitive_path in self._black_and_white_lists.get('sensitive_paths', []):
                if sensitive_path in path_in_args:
                    return True, f'Sensitive path "{sensitive_path}" detected'
            if path.suffix not in self._black_and_white_lists['safe_extensions']:
                return True, f'Unusual extension "{path.suffix}"'
        if name == 'local_python_interpreter':
            return self._analyze_python_code(args.get('code', ''))
        return False, None

    def invoke(self, name, args, task_id):
        with self._io_lock:
            current_ws = CWD_VAR.get()
            console.print(
                Panel(f'Agent Request: [bold cyan]{name}[/]\nWorkspace: [dim]{current_ws}[/]', border_style='blue')
            )

            if 'code' in args:
                console.print(Syntax(args['code'], 'python', theme='monokai', line_numbers=True))
            elif 'command' in args:
                console.print(Syntax(args['command'], 'bash', theme='monokai'))
            else:
                console.print(str(args))

            is_sensitive, reason = self._is_sensitive(name, args)

            if self.auto_mode and not is_sensitive:
                console.print(f'[bold green]⚡ Auto-Executing Task {task_id} (Safe)[/]')
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


SHARED_GUARD = InteractiveToolGuard()


class RecursiveToolAgent(Runnable):
    def __init__(self, agent, schema: Type[BaseModel], tools: List[Any], work_dir: str = None, max_recursion_depth=50):
        self.agent = agent
        self.schema = schema
        self.tools_map = {t.name: t for t in tools}
        self.model_with_tools = self.agent.bind_tools(tools+[schema])
        self.guard = SHARED_GUARD
        self.work_dir = work_dir
        self.max_recursion_depth = max_recursion_depth

    def invoke(self, input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        token = None
        if self.work_dir:
            token = CWD_VAR.set(self.work_dir)
        try:
            if isinstance(input, str):
                messages = [HumanMessage(content=input)]
            elif isinstance(input, dict):
                messages = [HumanMessage(content=str(input))]
            else:
                messages = list(input)
            recursion_count = 0
            while True:
                result = self.model_with_tools.invoke(messages, config=config)
                messages.append(result)
                if not result.tool_calls:
                    try:
                        return self.agent.with_structured_output(self.schema).invoke(messages)
                    except:
                        return None
                outputs = []
                is_finished = None
                for call in result.tool_calls:
                    name, args, call_id = call['name'], call['args'], call['id']
                    if name == self.schema.__name__:
                        try:
                            return self.schema(**args)
                        except:
                            return
                    if name in self.tools_map:
                        output = self.guard.invoke(name, args, call_id)
                        outputs.append(ToolMessage(content=str(output), tool_call_id=call_id, name=name))
                if is_finished:
                    return is_finished
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
                CWD_VAR.reset(token)


class CodeAgentBuilder:
    def __init__(self, model_id, tools=None, work_dir=None, max_recursion_depth=50, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.tools = tools or []
        self.work_dir = work_dir
        self.max_recursion_depth = max_recursion_depth
        self._agent = init_chat_model(self.model_id, **kwargs)

    @property
    def agent(self):
        return self._agent

    def with_structured_output(self, schema: Type[BaseModel]):
        if self.tools:
            return RecursiveToolAgent(
                self.agent,
                schema,
                tools=self.tools,
                work_dir=self.work_dir,
                max_recursion_depth=self.max_recursion_depth
            )
        return self.agent.with_structured_output(schema)

    def invoke(self, *args, **kwargs):
        return self.agent.invoke(*args, **kwargs)


def build_model(model_id, gcri_options=None, work_dir=None, **parameters):
    if gcri_options is not None:
        use_code_tools = gcri_options.get('use_code_tools', False)
        use_web_search = gcri_options.get('use_web_search', False)
        max_recursion_depth = gcri_options.get('max_recursion_depth', None)
    else:
        use_code_tools = False
        use_web_search = False
        max_recursion_depth = None
    tools = CLI_TOOLS if use_code_tools else []
    tools += [search_web] if use_web_search else []
    return CodeAgentBuilder(
        model_id,
        tools=tools,
        work_dir=work_dir,
        max_recursion_depth=max_recursion_depth,
        **parameters
    )
