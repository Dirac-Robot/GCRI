from inspect_ai import Task, task

try:
    from inspect_evals.swe_bench import swe_bench
except ImportError:
    swe_bench = None

from gcri.benchmark.solver import gcri_solver


@task
def gcri_swebench(
    endpoint: str = 'http://localhost:8001',
    timeout: float = 3600.0,
    dataset: str = 'princeton-nlp/SWE-bench_Verified',
    split: str = 'test',
    instance_ids: list = None
) -> Task:
    if swe_bench is None:
        raise ImportError('inspect_evals.swe_bench is not installed')
    solver = gcri_solver(endpoint=endpoint, timeout=timeout, benchmark_name='SWEBench')
    kwargs = {'solver': solver, 'dataset': dataset, 'split': split}
    if instance_ids:
        kwargs['instance_ids'] = instance_ids
    return swe_bench(**kwargs)
