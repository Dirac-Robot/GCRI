import json
import multiprocessing
import os

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from gcri.graphs.gcri_unit import GCRI
from gcri.config import scope

DATASET_HF_ID = "livecodebench/code_generation"
BENCHMARK_DIR = 'benchmark_results/livecodebench'
TIMEOUT_SECONDS = 15.


@scope
def get_preset_name(config):
    if config.get('custom_config_path'):
        return os.path.splitext(os.path.basename(config.custom_config_path))[0]
    return 'none'


RESULT_FILE = os.path.join(BENCHMARK_DIR, f'livecodebench_results_{get_preset_name()}.json')


class LiveCodeBenchResult(BaseModel):
    thought_process: str = Field(
        ...,
        description='Detailed reasoning about the algorithm, data structures, and edge cases.'
    )
    solution_code: str = Field(
        ...,
        description='The complete, executable Python code implementation only. No markdown formatting.'
    )


def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


def preprocess_code(code_str: str) -> str:
    if not code_str:
        return ''

    code_str = code_str.strip()

    if code_str.startswith('```python'):
        code_str = code_str[9:]
    elif code_str.startswith('```py'):
        code_str = code_str[5:]
    elif code_str.startswith('```'):
        code_str = code_str[3:]

    if code_str.endswith('```'):
        code_str = code_str[:-3]

    code_str = code_str.strip()
    code_str = code_str.replace('\\"', '"')
    code_str = code_str.replace("\\'", "'")

    lines = code_str.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.rstrip().endswith('\\'):
            line = line.rstrip()
        cleaned_lines.append(line)

    code_str = '\n'.join(cleaned_lines)
    return code_str


def run_test_case(test_program, input_data, result_queue):
    import sys
    import builtins
    from io import StringIO
    try:
        captured_output = StringIO()
        input_lines = input_data.strip().split('\n') if input_data else []
        input_iter = iter(input_lines)

        def mock_input(prompt=''):
            try:
                return next(input_iter)
            except StopIteration:
                return ''

        def mock_print(*args, **kwargs):
            kwargs['file'] = captured_output
            builtins._original_print(*args, **kwargs)

        builtins._original_print = builtins.print
        builtins.input = mock_input
        builtins.print = mock_print

        exec_globals = {'input': mock_input, 'print': mock_print}
        exec(test_program, exec_globals)

        output = captured_output.getvalue()
        result_queue.put(('output', output.strip()))
    except Exception as e:
        result_queue.put(('error', f'{type(e).__name__}: {str(e)}'))
    finally:
        builtins.print = builtins._original_print if hasattr(builtins, '_original_print') else print


def evaluate_code(sample, completion_code):
    common_imports = (
        'import sys\n'
        'import os\n'
        'import math\n'
        'import string\n'
        'import re\n'
        'import collections\n'
        'import heapq\n'
        'import itertools\n'
        'import functools\n'
        'import copy\n'
        'import bisect\n'
        'from typing import *\n\n'
    )

    public_tests_str = sample.get('public_test_cases', '[]')
    private_tests_str = sample.get('private_test_cases', '[]')

    try:
        public_tests = json.loads(public_tests_str) if isinstance(public_tests_str, str) else public_tests_str
    except json.JSONDecodeError:
        public_tests = []

    try:
        private_tests = json.loads(private_tests_str) if isinstance(private_tests_str, str) else private_tests_str
    except json.JSONDecodeError:
        private_tests = []

    all_tests = public_tests+private_tests

    if not all_tests:
        return False, 'No test cases available'

    full_code = common_imports+completion_code

    passed_count = 0
    total_tests = len(all_tests)

    for test_case in all_tests[:10]:
        if isinstance(test_case, dict):
            input_data = test_case.get('input', '')
            expected_output = test_case.get('output', '').strip()
        elif isinstance(test_case, (list, tuple)) and len(test_case) >= 2:
            input_data = test_case[0]
            expected_output = str(test_case[1]).strip()
        else:
            continue

        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=run_test_case,
            args=(full_code, input_data, result_queue)
        )

        process.start()
        process.join(TIMEOUT_SECONDS)

        if process.is_alive():
            process.terminate()
            process.join()
            return False, 'Timeout'

        if not result_queue.empty():
            status, result = result_queue.get()
            if status == 'error':
                return False, result
            if result != expected_output:
                return False, f'Wrong answer: expected "{expected_output[:50]}", got "{result[:50]}"'
            passed_count += 1
        else:
            return False, 'No result (Process crashed)'

    if passed_count == min(10, total_tests):
        return True, f'Passed ({passed_count}/{total_tests} tests)'
    else:
        return False, f'Failed ({passed_count}/{total_tests} tests)'


@scope
def run_benchmark(config, num_samples=None, split='test'):
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()

    logger.info('🤖 GCRI Worker Initializing for LiveCodeBench...')
    worker = GCRI(config, schema=LiveCodeBenchResult)

    logger.info(f'📚 Loading LiveCodeBench dataset from {DATASET_HF_ID} (split: {split})...')
    try:
        dataset = load_dataset(DATASET_HF_ID, split=split)
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

    logger.info(f'📊 Loaded {len(dataset)} tasks')

    if num_samples:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
        logger.info(f'🔍 Running on first {num_samples} samples.')

    results = []
    processed_ids = set()
    total_processed = 0
    total_passed = 0

    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                valid_results = []
                for item in existing_data:
                    t_id = item.get('question_id')
                    comp = item.get('completion')
                    error_log = item.get('error_log', '')
                    if not (comp and isinstance(comp, str) and comp.strip()):
                        logger.info(f'♻️ Re-queueing Task {t_id} (Reason: Empty completion)')
                    elif 'No module named' in error_log:
                        logger.info(f'♻️ Re-queueing Task {t_id} (Reason: {error_log})')
                    else:
                        valid_results.append(item)
                        processed_ids.add(t_id)
                results = valid_results
                total_processed = len(results)
                total_passed = sum(1 for item in results if item.get('passed', False))
                logger.info(f'🔄 Resuming... {total_processed} valid items retained.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='LiveCodeBench'):
        question_id = item.get('question_id', str(idx))
        if question_id in processed_ids:
            continue

        try:
            question_title = item.get('question_title', '')
            question_content = item.get('question_content', '')
            starter_code = item.get('starter_code', '')
            difficulty = item.get('difficulty', 'unknown')
            platform = item.get('platform', '')

            task_prompt = (
                f'You are an expert competitive programmer.\n'
                f'Solve the following programming problem from {platform}.\n\n'
                f'--- PROBLEM: {question_title} ---\n'
                f'Difficulty: {difficulty}\n\n'
                f'{question_content}\n\n'
            )

            if starter_code:
                task_prompt += f'--- STARTER CODE ---\n{starter_code}\n\n'

            task_prompt += (
                f'Provide a complete Python solution that:\n'
                f'1. Reads input from stdin\n'
                f'2. Prints output to stdout\n'
                f'3. Handles all edge cases\n\n'
                f'Include your reasoning first, then the complete solution code.'
            )

            logger.info(f'▶ Running Task: {question_id} ({question_title[:30]}...)')
            output_state = worker(task_prompt, commit_mode='auto-reject')
            final_output_obj = output_state.get('final_output')

            parsed_code = ''
            parsed_reasoning = ''

            if final_output_obj:
                if isinstance(final_output_obj, dict):
                    raw_code = final_output_obj.get('solution_code', '')
                    parsed_code = preprocess_code(raw_code)
                    parsed_reasoning = final_output_obj.get('thought_process', '')
                    raw_dump = final_output_obj
                else:
                    raw_dump = str(final_output_obj)
                    parsed_code = preprocess_code(str(final_output_obj))
            else:
                raw_dump = 'No final output generated.'

            is_passed, eval_message = evaluate_code(item, parsed_code)

            total_processed += 1
            if is_passed:
                total_passed += 1

            current_accuracy = (total_passed/total_processed)*100
            logger.info(
                f'🧪 Result: {"✅ PASS" if is_passed else "❌ FAIL"} ({eval_message}) | Acc: {current_accuracy:.2f}%'
            )

            result = {
                'question_id': question_id,
                'question_title': question_title,
                'platform': platform,
                'difficulty': difficulty,
                'prompt': question_content[:1000],
                'completion': parsed_code,
                'reasoning': parsed_reasoning,
                'passed': is_passed,
                'error_log': eval_message,
                'raw_output': raw_dump,
                'full_state': {
                    'best_branch': output_state.get('best_branch_index'),
                    'decision': output_state.get('decision'),
                    'iterations': output_state.get('count', 0)
                }
            }
            results.append(result)

            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except KeyboardInterrupt:
            logger.warning('⛔ Benchmark interrupted by user.')
            break
        except Exception as e:
            logger.error(f'❌ Error processing sample {question_id}: {e}')
            continue

    final_acc = (total_passed/len(dataset))*100 if len(dataset) > 0 else 0
    logger.info(f'✅ LiveCodeBench completed. Final Accuracy: {final_acc:.2f}%')
    logger.info(f'📄 Detailed results saved to {RESULT_FILE}')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_benchmark()
