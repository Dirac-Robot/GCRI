from inspect_ai import Task, task
from inspect_evals.bigcodebench import bigcodebench
from gcri.benchmark.solver import gcri_solver


@task
def gcri_bigcodebench(
    endpoint: str = 'http://localhost:8001',
    timeout: float = 300.0,
    sandbox: str = 'docker',
    version: str = 'v0.1.2'
) -> Task:
    """
    Run BigCodeBench evaluation using GCRI agent.
    
    Args:
        endpoint: GCRI benchmark server endpoint.
        timeout: Timeout for each solver call.
        sandbox: Sandbox type for execution (default: 'docker').
        version: BigCodeBench version.
    """
    # Create GCRI solver configured with endpoint
    solver = gcri_solver(endpoint=endpoint, timeout=timeout)
    
    # Return the InspectEval bigcodebench task with our solver injected
    return bigcodebench(
        solver=solver,
        sandbox=sandbox,
        version=version
    )
