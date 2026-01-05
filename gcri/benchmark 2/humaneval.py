from inspect_ai import Task, task
from inspect_evals.humaneval import humaneval
from gcri.benchmark.solver import gcri_solver


@task
def gcri_humaneval(
    endpoint: str = 'http://localhost:8001',
    timeout: float = 1800.0,
    sandbox: str = 'docker'
) -> Task:
    solver = gcri_solver(endpoint=endpoint, timeout=timeout, benchmark_name='HumanEval')
    return humaneval(solver=solver, sandbox=sandbox)
