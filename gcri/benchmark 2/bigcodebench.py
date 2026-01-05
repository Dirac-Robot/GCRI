from inspect_ai import Task, task
from inspect_evals.bigcodebench import bigcodebench
from gcri.benchmark.solver import gcri_solver


@task
def gcri_bigcodebench(
    endpoint: str = 'http://localhost:8001',
    timeout: float = 1800.0,
    sandbox: str = 'docker',
    version: str = 'v0.1.2'
) -> Task:
    solver = gcri_solver(endpoint=endpoint, timeout=timeout, benchmark_name='BigCodeBench')
    return bigcodebench(solver=solver, sandbox=sandbox, version=version)
