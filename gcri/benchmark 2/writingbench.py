from typing import Optional, List
from inspect_ai import Task, task

try:
    from inspect_evals.writingbench import writingbench
except ImportError:
    writingbench = None


@task
def gcri_writingbench(
    judge_model: str = 'openai/gpt-4o-mini',
    domain1: Optional[List[str]] = None,
    domain2: Optional[List[str]] = None
) -> Task:
    if writingbench is None:
        raise ImportError('inspect_evals.writingbench is not installed')
    kwargs = {'judge_model': judge_model}
    if domain1:
        kwargs['domain1'] = domain1
    if domain2:
        kwargs['domain2'] = domain2
    return writingbench(**kwargs)
