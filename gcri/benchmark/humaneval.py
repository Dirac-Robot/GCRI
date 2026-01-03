from inspect_ai import Task, task
from inspect_evals.humaneval import humaneval
from gcri.benchmark.solver import gcri_solver


@task
def gcri_humaneval(
    endpoint: str = 'http://localhost:8001',
    timeout: float = 300.0,
    sandbox: str = 'docker'
) -> Task:
    """
    Run HumanEval benchmark using GCRI agent.
    
    Args:
        endpoint: GCRI benchmark server endpoint.
        timeout: Timeout for each solver call.
        sandbox: Sandbox type for code execution (default: 'docker').
    """
    solver = gcri_solver(endpoint=endpoint, timeout=timeout)
    return humaneval(
        solver=solver,
        sandbox=sandbox
    )
