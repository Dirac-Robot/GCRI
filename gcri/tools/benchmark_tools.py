"""
InspectAI-based GCRI Benchmark Wrapper.

When running inside InspectAI's Docker sandbox, patches GCRI's sandbox
to avoid Docker-in-Docker conflicts.
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

try:
    from inspect_ai import Task, task
    from inspect_ai.dataset import Dataset, Sample
    from inspect_ai.model import ModelOutput
    from inspect_ai.scorer import Scorer, accuracy, includes
    from inspect_ai.solver import Solver, TaskState, solver
    from inspect_ai.util import sandbox
    INSPECT_AVAILABLE = True
except ImportError:
    INSPECT_AVAILABLE = False
    Solver = object
    TaskState = object
    sandbox = None

from ato import scope as config_scope
from gcri.graphs.gcri_unit import GCRI


class InspectSandboxAdapter:
    """Wraps InspectAI's SandboxEnvironment for GCRI compatibility."""

    def __init__(self):
        if not INSPECT_AVAILABLE or sandbox is None:
            raise ImportError('InspectAI sandbox is not available.')
        self._sandbox = None

    async def get_sandbox(self):
        if self._sandbox is None:
            self._sandbox = sandbox()
        return self._sandbox

    async def execute_command(self, command: str) -> str:
        sb = await self.get_sandbox()
        result = await sb.exec(['bash', '-c', command])
        if result.success:
            return result.stdout
        return f'Error (exit {result.returncode}): {result.stderr}'

    async def execute_python(self, code: str) -> str:
        sb = await self.get_sandbox()
        result = await sb.exec(['python', '-c', code])
        if result.success:
            return result.stdout
        return f'Error (exit {result.returncode}): {result.stderr}'

    async def write_file(self, path: str, content: str) -> str:
        sb = await self.get_sandbox()
        await sb.write_file(path, content)
        return f'Successfully wrote to {path}'

    async def read_file(self, path: str) -> str:
        sb = await self.get_sandbox()
        return await sb.read_file(path)

    def cleanup(self):
        pass


@dataclass
class GCRIConfig:
    """GCRI configuration container for benchmark execution."""
    preset_path: str
    commit_mode: str = 'auto-reject'
    max_iterations: int = 3


def load_gcri_config(preset_path: str) -> Any:
    """Load GCRI configuration from a preset file."""
    config_scope.load(preset_path)
    return config_scope.view


def _ensure_inspect():
    if not INSPECT_AVAILABLE:
        raise ImportError(
            'InspectAI is not installed. Please install it with: '
            'pip install inspect-ai'
        )


def detect_inspect_sandbox() -> bool:
    """Detect if running inside InspectAI's Docker sandbox."""
    indicators = [
        os.environ.get('INSPECT_SANDBOX'),
        os.path.exists('/.inspect_sandbox'),
        os.environ.get('INSPECT_EVAL_ID'),
    ]
    return any(indicators)


if INSPECT_AVAILABLE:
    @solver
    def gcri_solver(
        preset_path: str,
        commit_mode: str = 'auto-reject',
        schema: type[BaseModel] | None = None,
        use_inspect_sandbox: bool | None = None
    ) -> Solver:
        """Wrap GCRI as an InspectAI Solver."""
        config = load_gcri_config(preset_path)

        async def solve(state: TaskState, generate) -> TaskState:
            should_use_inspect_sandbox = use_inspect_sandbox
            if should_use_inspect_sandbox is None:
                should_use_inspect_sandbox = detect_inspect_sandbox()
            if should_use_inspect_sandbox:
                state = await _solve_with_inspect_sandbox(state, config, schema, commit_mode)
            else:
                state = await _solve_with_gcri_sandbox(state, config, schema, commit_mode)
            return state

        return solve

    async def _solve_with_inspect_sandbox(
        state: TaskState,
        config,
        schema: type[BaseModel] | None,
        commit_mode: str
    ) -> TaskState:
        """Execute GCRI with InspectAI sandbox (patches DockerSandbox at runtime)."""
        import asyncio
        import shutil
        import gcri.tools.docker_sandbox as docker_sandbox_module
        original_get_docker_sandbox = docker_sandbox_module.get_docker_sandbox
        adapter = InspectSandboxAdapter()

        def async_to_sync(awaitable):
            """Safely execute async code from sync context."""
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, awaitable)
                    return future.result()
            return asyncio.run(awaitable)

        class InspectSandboxBridge:
            """Drop-in replacement for DockerSandbox that delegates to InspectAI."""

            def __init__(self, sandbox_config):
                self._adapter = adapter
                self._containers = {}
                self.timeout = getattr(sandbox_config.sandbox, 'timeout', 60)

            @property
            def docker_available(self):
                return True

            def setup_branch(self, iteration: int, branch: int, source_dir: str) -> str:
                container_id = f'inspect_branch_{iteration}_{branch}'
                self._containers[f'{iteration}_{branch}'] = container_id
                
                # Copy source files to sandbox
                self._copy_to_sandbox(source_dir)
                return container_id

            def _copy_to_sandbox(self, source_dir: str):
                """Recursively copy files to sandbox."""
                ignore = {'.git', '__pycache__', 'venv', 'env', 'node_modules', '.idea', '.vscode', '.gcri'}
                for root, dirs, files in os.walk(source_dir):
                    dirs[:] = [d for d in dirs if d not in ignore]
                    rel_path = os.path.relpath(root, source_dir)
                    if rel_path == '.':
                        rel_path = ''
                    
                    # Create directory
                    if rel_path:
                        async_to_sync(self._adapter.execute_command(f'mkdir -p {rel_path}'))

                    for file in files:
                        if file in ignore:
                            continue
                        file_path = os.path.join(root, file)
                        dest_path = os.path.join(rel_path, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            async_to_sync(self._adapter.write_file(dest_path, content))
                        except UnicodeDecodeError:
                            pass  # Skip binary files

            def execute_command(self, container_id: str, command: str) -> str:
                return async_to_sync(self._adapter.execute_command(command))

            def execute_python(self, container_id: str, code: str) -> str:
                return async_to_sync(self._adapter.execute_python(code))

            def commit_to_host(self, container_id: str, target_dir: str):
                """Read files from sandbox and write to host (Minimal implementation)."""
                pass

            def cleanup_container(self, container_id: str):
                pass

            def cleanup_all(self):
                self._containers.clear()

            def get_container(self, iteration: int, branch: int) -> str:
                return self._containers.get(f'{iteration}_{branch}')

        bridge_instance = None

        def patched_get_docker_sandbox(sandbox_config):
            nonlocal bridge_instance
            if bridge_instance is None:
                bridge_instance = InspectSandboxBridge(sandbox_config)
            return bridge_instance

        docker_sandbox_module.get_docker_sandbox = patched_get_docker_sandbox
        docker_sandbox_module._docker_sandbox_instance = None
        task_input = state.input_text if hasattr(state, 'input_text') else str(state.input)
        try:
            gcri_instance = GCRI(config, schema=schema)
            result = gcri_instance(task=task_input, commit_mode=commit_mode)
            final_output = result.get('final_output', '')
            state.output = ModelOutput.from_content(
                model='gcri-inspect-sandbox',
                content=final_output if final_output else 'No output generated.'
            )
        except Exception as error:
            state.output = ModelOutput.from_content(
                model='gcri-inspect-sandbox',
                content=f'Error: {error}'
            )
        finally:
            docker_sandbox_module.get_docker_sandbox = original_get_docker_sandbox
            docker_sandbox_module._docker_sandbox_instance = None
        return state

    async def _solve_with_gcri_sandbox(
        state: TaskState,
        config,
        schema: type[BaseModel] | None,
        commit_mode: str
    ) -> TaskState:
        """Execute GCRI with native Docker sandbox."""
        gcri_instance = GCRI(config, schema=schema)
        task_input = state.input_text if hasattr(state, 'input_text') else str(state.input)
        try:
            result = gcri_instance(task=task_input, commit_mode=commit_mode)
            final_output = result.get('final_output', '')
            state.output = ModelOutput.from_content(
                model='gcri',
                content=final_output if final_output else 'No output generated.'
            )
        except Exception as error:
            state.output = ModelOutput.from_content(
                model='gcri',
                content=f'Error: {error}'
            )
        finally:
            if hasattr(gcri_instance, 'sandbox') and gcri_instance.sandbox:
                gcri_instance.sandbox.cleanup()
        return state

    def make_gcri_task(
        dataset: Dataset,
        scorer: Scorer | list[Scorer],
        preset_path: str,
        commit_mode: str = 'auto-reject',
        schema: type[BaseModel] | None = None,
        name: str | None = None,
        use_inspect_sandbox: bool | None = None
    ) -> Task:
        """Create an InspectAI Task with GCRI solver."""
        return Task(
            dataset=dataset,
            plan=[gcri_solver(preset_path, commit_mode, schema, use_inspect_sandbox)],
            scorer=scorer if isinstance(scorer, list) else [scorer],
            name=name
        )

    def create_sample(
        input_text: str,
        target: str | list[str],
        sample_id: str | None = None,
        metadata: dict | None = None
    ) -> Sample:
        """Create an InspectAI Sample from benchmark data."""
        return Sample(
            input=input_text,
            target=target if isinstance(target, list) else [target],
            id=sample_id,
            metadata=metadata or {}
        )


class GCRIBenchmark(ABC):
    """Abstract base class for GCRI benchmarks."""

    def __init__(
        self,
        preset_path: str,
        commit_mode: str = 'auto-reject',
        use_inspect_sandbox: bool | None = None
    ):
        _ensure_inspect()
        self.preset_path = preset_path
        self.commit_mode = commit_mode
        self.use_inspect_sandbox = use_inspect_sandbox

    @abstractmethod
    def load_dataset(self) -> 'Dataset':
        pass

    @abstractmethod
    def create_scorer(self) -> 'Scorer | list[Scorer]':
        pass

    def get_schema(self) -> type[BaseModel] | None:
        return None

    def get_name(self) -> str:
        return self.__class__.__name__

    @task
    def make_task(self) -> 'Task':
        return make_gcri_task(
            dataset=self.load_dataset(),
            scorer=self.create_scorer(),
            preset_path=self.preset_path,
            commit_mode=self.commit_mode,
            schema=self.get_schema(),
            name=self.get_name(),
            use_inspect_sandbox=self.use_inspect_sandbox
        )


class SimpleQABenchmark(GCRIBenchmark):
    """SimpleQA-style benchmark implementation."""

    def __init__(
        self,
        preset_path: str,
        qa_data: list[dict],
        commit_mode: str = 'auto-reject',
        use_inspect_sandbox: bool | None = None
    ):
        super().__init__(preset_path, commit_mode, use_inspect_sandbox)
        self.qa_data = qa_data

    def load_dataset(self) -> 'Dataset':
        samples = [
            create_sample(
                input_text=item['question'],
                target=item['answer'],
                sample_id=str(index),
                metadata=item.get('metadata', {})
            )
            for index, item in enumerate(self.qa_data)
        ]
        return Dataset(samples=samples)

    def create_scorer(self) -> 'Scorer':
        return includes()


class CodeExecutionBenchmark(GCRIBenchmark):
    """
    Code execution benchmark (HumanEval-style).
    
    When use_inspect_sandbox=True, code execution uses InspectAI's
    Docker container instead of GCRI's nested containers.
    """

    def __init__(
        self,
        preset_path: str,
        problems: list[dict],
        commit_mode: str = 'auto-reject',
        use_inspect_sandbox: bool | None = None
    ):
        super().__init__(preset_path, commit_mode, use_inspect_sandbox)
        self.problems = problems

    def load_dataset(self) -> 'Dataset':
        samples = [
            create_sample(
                input_text=problem['prompt'],
                target='PASSED',
                sample_id=problem.get('task_id', str(index)),
                metadata={
                    'test_code': problem['test_code'],
                    'entry_point': problem.get('entry_point', 'solution')
                }
            )
            for index, problem in enumerate(self.problems)
        ]
        return Dataset(samples=samples)

    def create_scorer(self) -> 'Scorer':
        return accuracy()

    def get_name(self) -> str:
        return 'CodeExecutionBenchmark'


def run_benchmark(benchmark: GCRIBenchmark, **eval_kwargs) -> Any:
    """Execute a GCRI benchmark using InspectAI."""
    _ensure_inspect()
    from inspect_ai import eval as inspect_eval
    benchmark_task = benchmark.make_task()
    return inspect_eval(benchmark_task, **eval_kwargs)
