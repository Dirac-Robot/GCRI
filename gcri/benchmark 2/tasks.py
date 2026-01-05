from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, csv_dataset, Sample
from inspect_ai.scorer import exact, includes, match, model_graded_qa, model_graded_fact

from gcri.benchmark.solver import gcri_solver


ScorerType = Literal['exact', 'includes', 'match', 'model_graded_qa', 'model_graded_fact']


def get_scorer(scorer_type: ScorerType):
    scorers = {
        'exact': exact(),
        'includes': includes(),
        'match': match(),
        'model_graded_qa': model_graded_qa(),
        'model_graded_fact': model_graded_fact()
    }
    return scorers.get(scorer_type, exact())


@task
def gcri_benchmark(
    dataset_path: str,
    scorer_type: ScorerType = 'exact',
    endpoint: str = 'http://localhost:8001',
    timeout: float = 300.0
) -> Task:
    if dataset_path.endswith('.csv'):
        dataset = csv_dataset(dataset_path)
    else:
        dataset = json_dataset(dataset_path)
    return Task(
        dataset=dataset,
        solver=gcri_solver(endpoint=endpoint, timeout=timeout),
        scorer=get_scorer(scorer_type)
    )


@task
def gcri_qa_benchmark(
    dataset_path: str,
    endpoint: str = 'http://localhost:8001'
) -> Task:
    return gcri_benchmark(
        dataset_path=dataset_path,
        scorer_type='model_graded_qa',
        endpoint=endpoint
    )


@task
def gcri_exact_benchmark(
    dataset_path: str,
    endpoint: str = 'http://localhost:8001'
) -> Task:
    return gcri_benchmark(
        dataset_path=dataset_path,
        scorer_type='exact',
        endpoint=endpoint
    )
