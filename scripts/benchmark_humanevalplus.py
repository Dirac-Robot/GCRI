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

DATASET_HF_ID = "evalplus/humanevalplus"
BENCHMARK_DIR = 'benchmark_results/humanevalplus'
TIMEOUT_SECONDS = 5.


@scope
def get_preset_name(config):
    if config.get('custom_config_path'):
        return os.path.splitext(os.path.basename(config.custom_config_path))[0]
    return 'none'


RESULT_FILE = os.path.join(BENCHMARK_DIR, f'humanevalplus_results_{get_preset_name()}.json')


class HumanEvalPlusResult(BaseModel):
    thought_process: str = Field(
        ...,
        description='Detailed reasoning about the algorithm and edge cases.'
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

    lines = code_str.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.rstrip().endswith('\\'):
            line = line.rstrip()
        cleaned_lines.append(line)

    code_str = '\n'.join(cleaned_lines)

    return code_str


def run_test_case(test_program, result_queue):
    try:
        exec_globals = {}
        exec(test_program, exec_globals)
        result_queue.put('passed')
    except AssertionError as e:
        result_queue.put(f'assertion_failed: {str(e)}')
    except Exception as e:
        result_queue.put(f'failed: {type(e).__name__}: {str(e)}')


def evaluate_code(sample, completion_code):
    header = (
        'import math\n'
        'import string\n'
        'import re\n'
        'import collections\n'
        'import heapq\n'
        'import itertools\n'
        'import functools\n'
        'import sys\n'
        'import copy\n'
        'import hashlib\n'
        'from typing import *\n\n'
    )

    prompt = sample['prompt']
    entry_point = sample['entry_point']
    plus_input = sample.get('plus_input', sample.get('plus', {}).get('input', []))
    contract = sample.get('contract', '')

    full_code = header+prompt+'\n'+completion_code+'\n\n'

    if contract:
        full_code += contract+'\n\n'

    test_cases = []
    if plus_input:
        for test_input in plus_input:
            if isinstance(test_input, (list, tuple)):
                args_str = ', '.join(repr(arg) for arg in test_input)
            else:
                args_str = repr(test_input)
            test_cases.append(f'{entry_point}({args_str})')

    if not test_cases:
        base_test = sample.get('test', '')
        if base_test:
            full_code += base_test+'\n\n'
            full_code += f'check({entry_point})'
        else:
            return False, 'No tests available'
    else:
        for tc in test_cases[:50]:
            full_code += f'_ = {tc}\n'

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_test_case, args=(full_code, result_queue))

    process.start()
    process.join(TIMEOUT_SECONDS)

    if process.is_alive():
        process.terminate()
        process.join()
        return False, 'Timeout'

    if not result_queue.empty():
        result = result_queue.get()
        if result == 'passed':
            return True, 'Passed'
        else:
            return False, result
    else:
        return False, 'No result (Process crashed)'


@scope
def run_benchmark(config, num_samples=None):
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()

    logger.info('🤖 GCRI Worker Initializing for HumanEval+...')
    worker = GCRI(config, schema=HumanEvalPlusResult)

    logger.info(f'📚 Loading HumanEval+ dataset from {DATASET_HF_ID}...')
    try:
        dataset = load_dataset(DATASET_HF_ID, split='test')
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

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
                    t_id = item.get('task_id')
                    comp = item.get('completion')
                    if comp and isinstance(comp, str) and comp.strip():
                        valid_results.append(item)
                        processed_ids.add(t_id)
                    else:
                        logger.info(f'♻️ Re-queueing Task {t_id} (Reason: Empty completion)')
                results = valid_results
                total_processed = len(results)
                total_passed = sum(1 for item in results if item.get('passed', False))
                logger.info(f'🔄 Resuming... {total_processed} valid items retained.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='HumanEval+ Benchmark'):
        task_id = item.get('task_id')
        if task_id in processed_ids:
            continue

        try:
            function_prompt = item.get('prompt', '')
            task_prompt = (
                f'You are an expert Python software engineer.\n'
                f'Complete the following Python function based on the provided signature and docstring.\n'
                f'Your code must be valid Python and strictly follow the indentation.\n'
                f'Pay special attention to edge cases as this benchmark has extensive test coverage.\n\n'
                f'{function_prompt}\n\n'
                f'Provide the reasoning and the fully functional code implementation.'
            )

            logger.info(f'▶ Running Task: {task_id}')
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
                f'🧪 Result: {'✅ PASS' if is_passed else '❌ FAIL'} ({eval_message}) | Acc: {current_accuracy:.2f}%'
            )

            result = {
                'task_id': task_id,
                'prompt': function_prompt,
                'canonical_solution': item.get('canonical_solution'),
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
            logger.error(f'❌ Error processing sample {task_id}: {e}')
            continue

    final_acc = (total_passed/len(dataset))*100 if len(dataset) > 0 else 0
    logger.info(f'✅ HumanEval+ Benchmark completed. Final Accuracy: {final_acc:.2f}%')
    logger.info(f'📄 Detailed results saved to {RESULT_FILE}')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_benchmark()
