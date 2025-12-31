from gcri.config import scope
from gcri.tools.benchmark_tools import run_benchmark, GCRIBenchmark, create_sample

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError('Please install datasets library: pip install datasets')

from inspect_ai.scorer import Scorer, Score, Target, accuracy, stderr, scorer
from inspect_ai.solver import TaskState

from loguru import logger


@scorer(metrics=[accuracy(), stderr()])
def verify_humaneval() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        code = state.output.completion
        if not code:
            return Score(value=0.0, explanation='No code generated')

        test_code = state.metadata.get('test_code', '')
        entry_point = state.metadata.get('entry_point', '')

        full_code = f'{code}\n\n{test_code}\n\ncheck({entry_point})'
        from inspect_ai.util import sandbox

        try:
            sb = sandbox()
            filename = 'solution.py'
            await sb.write_file(filename, full_code)

            result = await sb.exec(['python3', filename], timeout=10)

            if result.success:
                return Score(value=1.0, explanation='Tests passed')
            else:
                return Score(
                    value=0.0,
                    explanation=f'Tests failed. Exit code: {result.returncode}\nStderr: {result.stderr}\nStdout: '
                                f'{result.stdout}'
                )
        except Exception as e:
            return Score(value=0.0, explanation=f'Sandbox execution error: {str(e)}')

    return score


class HumanEvalBenchmark(GCRIBenchmark):
    def load_dataset(self) -> list:
        logger.info('Loading HumanEval dataset from HuggingFace...')
        hf_data = load_dataset('openai_humaneval', split='test')

        samples = []
        for item in hf_data:
            samples.append(
                create_sample(
                    input_text=item['prompt'],
                    target='PASSED',
                    sample_id=item['task_id'],
                    metadata={
                        'test_code': item['test'],
                        'entry_point': item['entry_point']
                    }
                )
            )

        return samples

    def create_scorer(self) -> Scorer:
        return verify_humaneval()

    def get_name(self) -> str:
        return 'humaneval'


@scope.observe(default=True)
def benchmark(config):
    config.limit = 0
    config.use_inspect_sandbox = False


@scope
def main(config):
    if not config.custom_config_path:
        logger.info('Error: Preset path required. Usage: python script.py preset:=path/to/preset.yaml')
        return

    benchmark = HumanEvalBenchmark(
        preset_path=config.preset,
        use_inspect_sandbox=True if config.use_inspect_sandbox else None
    )

    logger.info(f'Starting HumanEval Benchmark with preset: {config.preset}')
    full_dataset = benchmark.load_dataset()

    if config.limit > 0:
        limit_value = int(config.limit)
        sliced_samples = full_dataset[:limit_value]
        benchmark.load_dataset = lambda: sliced_samples

    results = run_benchmark(benchmark, log_level='info')

    logger.info('\nBenchmark Finished.')
    if results and results[0].results:
        scores = results[0].results.scores
        logger.info(f'Results: {scores}')


if __name__ == '__main__':
    main()
