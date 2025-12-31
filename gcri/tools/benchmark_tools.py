"""
InspectAI-based GCRI Benchmark Wrapper.

When running inside InspectAI's Docker sandbox, patches GCRI's sandbox
to avoid Docker-in-Docker conflicts.
"""
import concurrent.futures
import os
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
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

from gcri.graphs.gcri_unit import GCRI


class InspectSandboxAdapter:
    """Wraps InspectAI's SandboxEnvironment for GCRI compatibility."""

    def __init__(self, sandbox_instance=None):
        if not INSPECT_AVAILABLE or sandbox is None:
            raise ImportError('InspectAI sandbox is not available.')
        self._sandbox = sandbox_instance

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


def _ensure_inspect():
    if not INSPECT_AVAILABLE:
        raise ImportError(
            'InspectAI is not installed. Please install it with: '
            'pip install inspect-ai'
        )
    logger.info('InspectAI sandbox is available.')


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
        config,
        commit_mode: str = 'auto-reject',
        schema: type[BaseModel] | None = None,
        use_inspect_sandbox: bool | None = None
    ) -> Solver:
        """Wrap GCRI as an InspectAI Solver."""

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
        import gcri.tools.docker_sandbox as docker_sandbox_module

        logger.info('Benchmark is running on InspectAI sandbox now.')
        original_get_docker_sandbox = docker_sandbox_module.get_docker_sandbox

        # Fetch sandbox in the correct async context FIRST
        sb = sandbox()
        adapter = InspectSandboxAdapter(sb)

        # Reuse executor to avoid creation overhead
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        def async_to_sync(awaitable):
            """Safely execute async code from sync context."""
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                future = executor.submit(asyncio.run, awaitable)
                return future.result()
            return asyncio.run(awaitable)

        class InspectSandboxBridge:
            """Drop-in replacement for DockerSandbox that delegates to InspectAI."""
            _copied_dirs = set()  # Cache to avoid redundant file copies

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

                # Only copy files once per source directory
                if source_dir not in InspectSandboxBridge._copied_dirs:
                    self._copy_to_sandbox(source_dir)
                    InspectSandboxBridge._copied_dirs.add(source_dir)
                return container_id

            def _copy_to_sandbox(self, source_dir: str):
                """Copy files to sandbox using tar archive for bulk transfer."""
                import io
                import tarfile
                import base64

                ignore = {'.git', '__pycache__', 'venv', 'env', 'node_modules', '.idea', '.vscode', '.gcri'}

                # Create tar archive in memory
                tar_buffer = io.BytesIO()
                with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
                    for root, dirs, files in os.walk(source_dir):
                        dirs[:] = [d for d in dirs if d not in ignore]
                        for file in files:
                            if file in ignore:
                                continue
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, source_dir)
                            try:
                                tar.add(file_path, arcname=rel_path)
                            except (PermissionError, OSError):
                                pass

                # Encode and transfer as single file
                tar_data = base64.b64encode(tar_buffer.getvalue()).decode('ascii')
                async_to_sync(self._adapter.write_file('/tmp/workspace.tar.gz.b64', tar_data))

                # Decode and extract in container
                async_to_sync(
                    self._adapter.execute_command(
                        'cd /workspace && base64 -d /tmp/workspace.tar.gz.b64 | tar xzf - && rm '
                        '/tmp/workspace.tar.gz.b64'
                    )
                )

            def execute_command(self, container_id: str, command: str) -> str:
                return async_to_sync(self._adapter.execute_command(command))

            def execute_python(self, container_id: str, code: str) -> str:
                return async_to_sync(self._adapter.execute_python(code))

            def _execute_in_container(self, container_id: str, cmd_list: list) -> str:
                """Execute command list in container (compatibility with DockerSandbox API)."""
                command = ' '.join(cmd_list)
                return async_to_sync(self._adapter.execute_command(command))

            def commit_to_host(self, container_id: str, target_dir: str):
                """Read files from sandbox and write to host (Minimal implementation)."""
                pass

            def cleanup_container(self, container_id: str):
                pass

            def cleanup_all(self):
                self._containers.clear()

            def get_container(self, iteration: int, branch: int) -> str:
                return self._containers.get(f'{iteration}_{branch}')

        def patched_get_docker_sandbox(sandbox_config):
            # Always return a new instance to avoid state sharing between concurrent tasks
            return InspectSandboxBridge(sandbox_config)

        # Patch both module-level reference AND the utils.py local reference
        import gcri.tools.utils as utils_module
        original_utils_get_docker_sandbox = utils_module.get_docker_sandbox
        docker_sandbox_module.get_docker_sandbox = patched_get_docker_sandbox
        utils_module.get_docker_sandbox = patched_get_docker_sandbox
        docker_sandbox_module._docker_sandbox_instance = None
        task_input = state.input_text if hasattr(state, 'input_text') else str(state.input)
        try:
            gcri_instance = GCRI(config, schema=schema)
            result = gcri_instance(task=task_input, commit_mode=commit_mode)
            final_output = result.get('final_output', '')
            if not final_output and result.get('decision'):
                final_output = result['decision'].get('action_input', '')
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
            utils_module.get_docker_sandbox = original_utils_get_docker_sandbox
            docker_sandbox_module._docker_sandbox_instance = None
        return state


    async def _solve_with_gcri_sandbox(
        state: TaskState,
        config,
        schema: type[BaseModel] | None,
        commit_mode: str
    ) -> TaskState:
        """Execute GCRI with native Docker sandbox."""
        logger.info('Benchmark is running on own GCRI sandbox now.')
        gcri_instance = GCRI(config, schema=schema)
        task_input = state.input_text if hasattr(state, 'input_text') else str(state.input)
        try:
            result = gcri_instance(task=task_input, commit_mode=commit_mode)
            final_output = result.get('final_output', '')
            if not final_output and result.get('decision'):
                final_output = result['decision'].get('action_input', '')
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
        config,
        dataset: Dataset,
        scorer: Scorer | list[Scorer],
        commit_mode: str = 'auto-reject',
        schema: type[BaseModel] | None = None,
        name: str | None = None,
        use_inspect_sandbox: bool | None = None
    ) -> Task:
        """Create an InspectAI Task with GCRI solver."""
        task_kwargs = {
            'dataset': dataset,
            'plan': [gcri_solver(config, commit_mode, schema, use_inspect_sandbox)],
            'scorer': scorer if isinstance(scorer, list) else [scorer],
            'name': name
        }
        # Add sandbox if using inspect sandbox
        if use_inspect_sandbox:
            task_kwargs['sandbox'] = 'docker'
        return Task(**task_kwargs)


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
        config,
        commit_mode: str = 'auto-reject',
        use_inspect_sandbox: bool | None = None
    ):
        _ensure_inspect()
        self.config = config
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
            config=self.config,
            dataset=self.load_dataset(),
            scorer=self.create_scorer(),
            commit_mode=self.commit_mode,
            schema=self.get_schema(),
            name=self.get_name(),
            use_inspect_sandbox=self.use_inspect_sandbox
        )


class SimpleQABenchmark(GCRIBenchmark):
    """SimpleQA-style benchmark implementation."""

    def __init__(
        self,
        config,
        qa_data: list[dict],
        commit_mode: str = 'auto-reject',
        use_inspect_sandbox: bool | None = None
    ):
        super().__init__(config, commit_mode, use_inspect_sandbox)
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
        return samples

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
        config,
        problems: list[dict],
        commit_mode: str = 'auto-reject',
        use_inspect_sandbox: bool | None = None
    ):
        super().__init__(config, commit_mode, use_inspect_sandbox)
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
        return samples

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
