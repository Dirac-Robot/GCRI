import httpx
from inspect_ai.solver import Solver, TaskState, Generate, solver


@solver
def gcri_solver(
    endpoint: str = 'http://localhost:8001',
    timeout: float = 300.0,
    benchmark_name: str = 'Benchmark'
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = state.input_text
        sample_id = str(state.sample_id) if state.sample_id else 'unknown'
        task_id = f'{benchmark_name}-Q{sample_id}'
        metadata = dict(state.metadata or {})
        metadata['benchmark_type'] = benchmark_name
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f'{endpoint}/benchmark/solve',
                json={
                    'task_id': task_id,
                    'prompt': prompt,
                    'metadata': metadata
                }
            )
            response.raise_for_status()
            result = response.json()
        state.output.completion = result.get('answer', '')
        return state
    return solve
