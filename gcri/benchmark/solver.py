from typing import Callable

import httpx
from inspect_ai.solver import Solver, TaskState, Generate, solver


@solver
def gcri_solver(
    endpoint: str = 'http://localhost:8001',
    timeout: float = 300.0
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = state.input_text
        task_id = str(state.sample_id) if state.sample_id else 'unknown'
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f'{endpoint}/benchmark/solve',
                json={
                    'task_id': task_id,
                    'prompt': prompt,
                    'metadata': state.metadata or {}
                }
            )
            response.raise_for_status()
            result = response.json()
        state.output.completion = result.get('answer', '')
        return state
    return solve
