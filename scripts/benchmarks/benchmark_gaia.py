import re
import string
from typing import Any

from gcri.config import scope
from gcri.tools.benchmark_tools import run_benchmark, GCRIBenchmark, create_sample

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError('Please install required libraries: pip install datasets')

from inspect_ai.scorer import Scorer, Score, Target, accuracy, stderr, scorer, CORRECT, INCORRECT
from inspect_ai.solver import TaskState

from loguru import logger


# GAIA Scorer implementation (based on official GAIA scorer)
def normalize_number_str(number_str: str) -> float:
    for char in ['$', '%', ',']:
        number_str = number_str.replace(char, '')
    try:
        return float(number_str)
    except ValueError:
        return float('inf')


def split_string(s: str, char_list: list[str] = [',', ';']) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r'\s', '', input_str)
    if remove_punct:
        translator = str.maketrans('', '', string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def is_float(element: Any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def question_scorer(model_answer: str, ground_truth: str) -> tuple[bool, str]:
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(model_answer)
        return (
            normalized_answer == float(ground_truth),
            f'Evaluated {model_answer} as a number.'
        )
    elif any(char in ground_truth for char in [',', ';']):
        gt_answers = split_string(ground_truth)
        model_answers = split_string(model_answer)
        if len(gt_answers) != len(model_answers):
            return (
                False,
                f'Evaluated {model_answer} as a list, returned False because lists have different lengths.'
            )
        comparisons = []
        for model_answer, gt_answer in zip(model_answers, gt_answers):
            if is_float(gt_answer):
                normalized_ma_elem = normalize_number_str(model_answer)
                comparisons.append(normalized_ma_elem == float(gt_answer))
            else:
                comparisons.append(
                    normalize_str(model_answer, remove_punct=False) == normalize_str(gt_answer, remove_punct=False)
                )
        return all(comparisons), f'Evaluated {model_answer} as a comma separated list.'
    else:
        return (
            normalize_str(model_answer) == normalize_str(ground_truth),
            f'Evaluated {model_answer} as a string.'
        )


@scorer(metrics=[accuracy(), stderr()])
def gaia_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        if not answer:
            return Score(value=INCORRECT, answer='', explanation='No answer generated')
        is_correct, explanation = question_scorer(
            model_answer=answer,
            ground_truth=target.text
        )
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=answer,
            explanation=explanation
        )

    return score


DEFAULT_INPUT_PROMPT = '''
Answer the following question. You need to use tools to find the information if necessary.

{file}

Question:
{question}
'''.strip()


@scope.observe(default=True)
def defaults(config):
    config.limit = 0
    config.use_inspect_sandbox = True
    config.subset = '2023_all'
    config.split = 'validation'


class GAIABenchmark(GCRIBenchmark):
    def __init__(
        self,
        config,
        subset: str = '2023_all',
        split: str = 'validation',
        commit_mode: str = 'auto-reject',
        use_inspect_sandbox: bool = None
    ):
        super().__init__(config, commit_mode, use_inspect_sandbox)

        valid_subsets = ['2023_all', '2023_level1', '2023_level2', '2023_level3']
        s_subset = str(subset).strip()

        if s_subset not in valid_subsets:
            logger.warning(f"Invalid subset '{subset}' provided. Defaulting to '2023_all'.")
            self.subset = '2023_all'
        else:
            self.subset = s_subset

        self.split = str(split) if split else 'validation'

    def load_dataset(self) -> list:
        logger.info(f'Loading GAIA dataset (subset={self.subset}, split={self.split})...')

        # Load directly from HuggingFace Hub (auto-caches)
        # Ensure subset is string
        hf_data = load_dataset(
            'gaia-benchmark/GAIA',
            str(self.subset),
            split=self.split
        )

        samples = []
        for record in hf_data:
            task_id = record['task_id']
            question = record['Question']
            final_answer = record.get('Final answer', '')

            # Check if file is associated
            file_name = record.get('file_name', '')
            file_prompt = ''
            if file_name:
                file_prompt = f'The following file is referenced in the question: {file_name}'

            input_text = DEFAULT_INPUT_PROMPT.format(file=file_prompt, question=question)

            samples.append(
                create_sample(
                    input_text=input_text,
                    target=final_answer,
                    sample_id=task_id,
                    metadata={
                        'level': record.get('Level', ''),
                        'file_name': file_name,
                        'annotator_metadata': record.get('Annotator Metadata', {})
                    }
                )
            )

        return samples

    def create_scorer(self) -> Scorer:
        return gaia_scorer()

    def get_name(self) -> str:
        return f'gaia_{self.subset}'


@scope
def main(config):
    logger.info(config.to_xyz())

    if not config.custom_config_path:
        logger.error('Error: Preset path required. Usage: python script.py custom_config_path:=path/to/preset.yaml')
        return

    benchmark = GAIABenchmark(
        config=config,
        subset=config.subset,
        split=config.split,
        use_inspect_sandbox=True  # GAIA requires sandbox
    )

    logger.info(f'Starting GAIA Benchmark (subset={config.subset}, split={config.split})')
    full_dataset = benchmark.load_dataset()

    if config.limit > 0:
        limit_value = int(config.limit)
        sliced_samples = full_dataset[:limit_value]
        benchmark.load_dataset = lambda: sliced_samples

    results = run_benchmark(benchmark, log_level='info')

    logger.info('Benchmark Finished.')
    if results and results[0].results:
        scores = results[0].results.scores
        logger.info(f'Results: {scores}')


if __name__ == '__main__':
    main()
