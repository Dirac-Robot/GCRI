from inspect_ai import Task, task
from inspect_evals.gaia import gaia
from gcri.benchmark.solver import gcri_solver


@task
def gcri_gaia(
    endpoint: str = 'http://localhost:8001',
    timeout: float = 600.0,
    subset: str = '2023_level1',
    split: str = 'validation',
    max_messages: int = 100
) -> Task:
    """
    Run GAIA benchmark using GCRI agent.
    
    Args:
        endpoint: GCRI benchmark server endpoint.
        timeout: Timeout for each solver call.
        subset: GAIA subset ('2023_all', '2023_level1', '2023_level2', '2023_level3').
        split: Dataset split ('test', 'validation').
        max_messages: Maximum messages per task.
    """
    solver = gcri_solver(endpoint=endpoint, timeout=timeout)
    return gaia(
        solver=solver,
        subset=subset,
        split=split,
        max_messages=max_messages
    )
