import ast
import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Type

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


@scope
def _get_black_and_white_lists(config):
    lists = ADict.from_file(config.templates.black_and_white_lists).to_dict()
    lists['safe_extensions'] = set(lists.get('safe_extensions', []))
    lists['sensitive_modules'] = set(lists.get('sensitive_modules', []))
    return lists


@tool
def execute_shell_command(command: str) -> str:
    """Executes a shell command."""
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            return res.stdout if res.stdout.strip() else '(Success, no output)'
        return f'Exit Code {res.returncode}:\n{res.stderr}'
    except subprocess.TimeoutExpired:
        return 'Error: Command timed out.'
    except Exception as e:
        return f'Error: {e}'


@tool
def read_file(filepath: str) -> str:
    """Reads the content of a file."""
    if not os.path.exists(filepath):
        return f'Error: File "{filepath}" not found.'
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f'Error: {e}'


@tool
def write_file(filepath: str, content: str) -> str:
    """Writes content to a file."""
    try:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f'Successfully wrote to "{filepath}".'
    except Exception as e:
        return f'Error: {e}'


class PythonREPL:
    def __init__(self):
        self.globals = {}
        self.locals = {}

    def run(self, command: str) -> str:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = my_out = io.StringIO()
        sys.stderr = my_err = io.StringIO()
        try:
            exec(command, self.globals, self.locals)
            out, err = my_out.getvalue(), my_err.getvalue()
            if err:
                return f'{out}\nError: {err}'
            return out if out else '(No output)'
        except Exception as e:
            return f'{my_out.getvalue()}\nExecution Error: {e}'
        finally:
            sys.stdout, sys.stderr = old_out, old_err


@tool
def local_python_interpreter(code: str) -> str:
    """Executes Python code locally."""
    repl = PythonREPL()
    return repl.run(code)


CLI_TOOLS = [execute_shell_command, read_file, write_file, local_python_interpreter]


class InteractiveToolGuard:
    def __init__(self):
        self.tools = {t.name: t for t in CLI_TOOLS}
        self.auto_mode = False
        try:
            self._bw_lists = _get_black_and_white_lists()
        except:
            self._bw_lists = {
                'sensitive_commands': [],
                'sensitive_paths': [],
                'sensitive_modules': set(),
                'safe_extensions': set()
            }

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
                        if name in self._bw_lists['sensitive_modules']:
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
            cmd = args.get('command', '').strip()
            for sens in self._bw_lists.get('sensitive_commands', []):
                if sens in cmd.split():
                    return True, f'Command "{sens}" detected'
        if name == 'write_file':
            path = args.get('filepath', '')
            p = Path(path)
            for sens in self._bw_lists.get('sensitive_paths', []):
                if sens in path:
                    return True, f'Sensitive path "{sens}" detected'
            if p.suffix not in self._bw_lists['safe_extensions']:
                return True, f'Unusual extension "{p.suffix}"'
        if name == 'local_python_interpreter':
            return self._analyze_python_code(args.get('code', ''))
        return False, None

    def invoke(self, name, args, tid):
        console.print(Panel(f'Agent Request: [bold cyan]{name}[/]', border_style='blue'))

        if 'code' in args:
            console.print(Syntax(args['code'], 'python', theme='monokai', line_numbers=True))
        elif 'command' in args:
            console.print(Syntax(args['command'], 'bash', theme='monokai'))
        else:
            console.print(str(args))

        is_sensitive, reason = self._is_sensitive(name, args)

        if self.auto_mode:
            if not is_sensitive:
                console.print(f'[bold green]⚡ Auto-Executing Task {tid} (Safe)[/]')
                try:
                    res = self.tools[name].invoke(args)
                    console.print(f'[dim]Result: {str(res)[:100]}...[/]')
                    return str(res)
                except Exception as e:
                    return f'Tool Error: {e}'
            else:
                console.print(f'[bold red]✋ Auto-Mode Paused: {reason}[/]')

        while True:
            console.print('[bold yellow]>> (y)es / (a)lways / (n)o / (e)dit : [/]', end='')
            sys.stdout.flush()
            choice = input().strip().lower()

            if choice == 'y':
                console.print('[dim]Executing...[/]')
                try:
                    res = self.tools[name].invoke(args)
                    console.print('[bold green]Done.[/]')
                    return str(res)
                except Exception as e:
                    return f'Tool Error: {e}'

            elif choice == 'a':
                console.print('[bold green]⚡ Auto-Mode Enabled.[/]')
                self.auto_mode = True
                try:
                    res = self.tools[name].invoke(args)
                    console.print('[bold green]Done.[/]')
                    return str(res)
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
                new_content = '\n'.join(lines)

                if not new_content.strip():
                    console.print('No input provided.')
                    continue

                if name == 'execute_shell_command':
                    args['command'] = new_content
                elif name == 'write_file':
                    args['content'] = new_content
                elif name == 'local_python_interpreter':
                    args['code'] = new_content
                else:
                    console.print('Edit not supported.')
                    continue

                console.print('[dim]Executing modified...[/]')
                try:
                    res = self.tools[name].invoke(args)
                    console.print('[bold green]Done.[/]')
                    return str(res)
                except Exception as e:
                    return f'Tool Error: {e}'


SHARED_GUARD = InteractiveToolGuard()


class RecursiveToolAgent(Runnable):
    def __init__(self, agent, schema: Type[BaseModel], tools: List[Any], max_recursion_depth=20):
        self.agent = agent
        self.schema = schema
        self.tools_map = {t.name: t for t in tools}
        self.model_with_tools = self.agent.bind_tools(tools+[schema])
        self.guard = SHARED_GUARD
        self.max_recursion_depth = max_recursion_depth

    def invoke(self, input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        elif isinstance(input, dict):
            messages = [HumanMessage(content=str(input))]
        else:
            messages = list(input)

        for _ in range(self.max_recursion_depth):
            res = self.model_with_tools.invoke(messages, config=config)
            messages.append(res)

            if not res.tool_calls:
                try:
                    return self.agent.with_structured_output(self.schema).invoke(messages)
                except:
                    return None

            outputs = []
            is_finished = None

            for call in res.tool_calls:
                name, args, cid = call['name'], call['args'], call['id']

                if name == self.schema.__name__:
                    try:
                        return self.schema(**args)
                    except:
                        return None

                if name in self.tools_map:
                    output = self.guard.invoke(name, args, cid)
                    outputs.append(ToolMessage(content=str(output), tool_call_id=cid, name=name))

            if is_finished:
                return is_finished
            messages.extend(outputs)

        raise ValueError('Max steps exceeded.')


class CodeAgentBuilder:
    def __init__(self, model_id, tools=None, max_recursion_depth=20, **kwargs):
        self.model_id = model_id
        self.tools = tools or []
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
                max_recursion_depth=self.max_recursion_depth
            )
        return self.agent.with_structured_output(schema)

    def invoke(self, *args, **kwargs):
        return self.agent.invoke(*args, **kwargs)


def build_model(model_id, gcri_options=None, **parameters):
    if gcri_options is not None:
        use_code_tools = gcri_options.get('use_code_tools', False)
        max_recursion_depth = gcri_options.get('max_recursion_depth', 20)
    else:
        use_code_tools = False
        max_recursion_depth = 20
    tools = CLI_TOOLS if use_code_tools else []
    return CodeAgentBuilder(model_id, tools=tools, max_recursion_depth=max_recursion_depth, **parameters)
