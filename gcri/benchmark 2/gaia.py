from inspect_ai import Task, task
from inspect_evals.gaia import gaia
from gcri.benchmark.solver import gcri_solver


@task
def gcri_gaia(
    endpoint: str = 'http://localhost:8001',
    timeout: float = 1800.0,
    subset: str = '2023_level1',
    split: str = 'validation',
    max_messages: int = 100
) -> Task:
    solver = gcri_solver(endpoint=endpoint, timeout=timeout, benchmark_name='GAIA')
    return gaia(solver=solver, subset=subset, split=split, max_messages=max_messages)
