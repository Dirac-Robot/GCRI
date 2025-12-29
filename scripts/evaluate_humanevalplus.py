"""
HumanEval+ Post-Evaluation Script
Re-evaluates existing result files with the corrected evaluation logic.
"""
import json
import multiprocessing
import os
import sys

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

DATASET_HF_ID = "evalplus/humanevalplus"


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


def evaluate_code(sample, completion_code, timeout=5.0):
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

    plus_data = sample.get('plus', {})
    plus_inputs = plus_data.get('input', [])
    plus_outputs = plus_data.get('output', [])
    atol = plus_data.get('atol', 0)

    full_code = header+prompt+'\n'+completion_code+'\n\n'

    contract = sample.get('contract', '')
    if contract:
        full_code += contract+'\n\n'

    has_plus_tests = plus_inputs and plus_outputs and len(plus_inputs) == len(plus_outputs)

    if has_plus_tests:
        full_code += 'def _deep_eq(a, b, atol=0):\n'
        full_code += '    if isinstance(a, float) and isinstance(b, float):\n'
        full_code += '        return abs(a-b) <= atol\n'
        full_code += '    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):\n'
        full_code += '        return len(a) == len(b) and all(_deep_eq(x, y, atol) for x, y in zip(a, b))\n'
        full_code += '    return a == b\n\n'

        for i, (inp, expected) in enumerate(zip(plus_inputs[:50], plus_outputs[:50])):
            if isinstance(inp, (list, tuple)):
                args_str = ', '.join(repr(arg) for arg in inp)
            else:
                args_str = repr(inp)
            full_code += f'_result_{i} = {entry_point}({args_str})\n'
            full_code += f'assert _deep_eq(_result_{i}, {repr(expected)}, atol={atol}), f"Test {i}: expected {repr(expected)}, got {{_result_{i}}}"\n'
    else:
        base_test = sample.get('test', '')
        if base_test:
            full_code += base_test+'\n\n'
            full_code += f'check({entry_point})'
        else:
            return False, 'No tests available'

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_test_case, args=(full_code, result_queue))

    process.start()
    process.join(timeout)

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


def run_post_eval(result_file: str, output_file: str = None):
    load_dotenv()

    if output_file is None:
        base_name = os.path.splitext(result_file)[0]
        output_file = f'{base_name}_reeval.json'

    logger.info(f'📄 Loading results from {result_file}...')
    if not os.path.exists(result_file):
        logger.error(f'Result file not found: {result_file}')
        return

    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    logger.info(f'📚 Loading HumanEval+ dataset...')
    dataset = load_dataset(DATASET_HF_ID, split='test')
    dataset_map = {item['task_id']: item for item in dataset}

    logger.info(f'🔄 Re-evaluating {len(results)} results...')

    reeval_results = []
    total_passed = 0
    prev_passed = 0

    for item in tqdm(results, desc='Post-Eval'):
        task_id = item.get('task_id')
        completion = item.get('completion', '')
        prev_result = item.get('passed', False)

        if prev_result:
            prev_passed += 1

        if not completion:
            reeval_results.append({
                **item,
                'reeval_passed': False,
                'reeval_error': 'Empty completion',
                'prev_passed': prev_result
            })
            continue

        sample = dataset_map.get(task_id)
        if not sample:
            logger.warning(f'⚠️ Task {task_id} not found in dataset')
            reeval_results.append({
                **item,
                'reeval_passed': False,
                'reeval_error': 'Task not in dataset',
                'prev_passed': prev_result
            })
            continue

        is_passed, eval_message = evaluate_code(sample, completion)

        if is_passed:
            total_passed += 1

        status = '✅' if is_passed else '❌'
        change = ''
        if prev_result and not is_passed:
            change = ' [FALSE POSITIVE FIXED]'
        elif not prev_result and is_passed:
            change = ' [RECOVERED]'

        logger.info(f'{status} {task_id}: {eval_message}{change}')

        reeval_results.append({
            **item,
            'reeval_passed': is_passed,
            'reeval_error': eval_message,
            'prev_passed': prev_result
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reeval_results, f, indent=4, ensure_ascii=False)

    n = len(results)
    logger.info('\n📊 Post-Evaluation Summary:')
    logger.info(f'   Previous Accuracy: {prev_passed/n*100:.2f}% ({prev_passed}/{n})')
    logger.info(f'   Corrected Accuracy: {total_passed/n*100:.2f}% ({total_passed}/{n})')
    logger.info(f'   Difference: {(total_passed-prev_passed)/n*100:+.2f}%')
    logger.info(f'📄 Results saved to {output_file}')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    if len(sys.argv) < 2:
        logger.error('Usage: python evaluate_humanevalplus.py <result_file.json>')
        sys.exit(1)
    run_post_eval(sys.argv[1])
