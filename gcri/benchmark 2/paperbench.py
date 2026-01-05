from typing import Optional, List, Union
from inspect_ai import Task, task

try:
    from inspect_evals.paperbench import paperbench
except ImportError:
    paperbench = None

from gcri.benchmark.solver import gcri_solver


@task
def gcri_paperbench(
    endpoint: str = 'http://localhost:8001',
    timeout: float = 21600.0,
    split: str = 'prod',
    paper_ids: Optional[Union[str, List[str]]] = None,
    submissions_dir: str = './submissions',
    max_time_in_hours: float = 6.0,
    message_limit: int = 100,
    token_limit: Optional[int] = None,
    sandbox_type: str = 'docker'
) -> Task:
    if paperbench is None:
        raise ImportError('inspect_evals.paperbench is not installed')
    solver = gcri_solver(endpoint=endpoint, timeout=timeout, benchmark_name='PaperBench')
    return paperbench(
        solver=solver,
        split=split,
        paper_ids=paper_ids,
        submissions_dir=submissions_dir,
        max_time_in_hours=max_time_in_hours,
        message_limit=message_limit,
        token_limit=token_limit,
        sandbox_type=sandbox_type
    )
