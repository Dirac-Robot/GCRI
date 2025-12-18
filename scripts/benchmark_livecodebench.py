import ast
import json
import multiprocessing
import os

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

# GCRI 모듈
from gcri.config import scope
from gcri.graphs.gcri_unit import GCRI

BENCHMARK_DIR = 'benchmark_results/livecodebench'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'lcb_results.json')
TIMEOUT_SECONDS = 5.


class LCBSolution(BaseModel):
    thought_process: str = Field(
        ...,
        description='Detailed algorithm design, time complexity analysis, and edge case consideration.'
    )
    solution_code: str = Field(
        ...,
        description='The complete Python code. Must be wrapped in a class named "Solution" with the required method, '
                    'or a standalone function as requested.'
    )


def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


def preprocess_code(code_str: str) -> str:
    code_str = code_str.strip()
    if code_str.startswith('```python'):
        code_str = code_str[9:]
    elif code_str.startswith('```'):
        code_str = code_str[3:]
    if code_str.endswith('```'):
        code_str = code_str[:-3]
    return code_str.strip()


def run_test_case_lcb(user_code, test_inputs, test_outputs, entry_point, result_queue):
    try:
        exec_globals = {}
        exec(
            'import math\nimport collections\nimport heapq\nimport itertools\nimport functools\nfrom typing import *',
            exec_globals
        )
        exec(user_code, exec_globals)
        if entry_point not in exec_globals:
            raise ValueError(f'Entry point "{entry_point}" not found in executed code.')
        func = exec_globals[entry_point]
        for test_input, test_output in zip(test_inputs, test_outputs):
            try:
                if isinstance(test_input, str):
                    arg = json.loads(test_input)
                else:
                    arg = test_input
                if isinstance(test_output, str):
                    expected = json.loads(test_output)
                else:
                    expected = test_output
            except:
                arg = eval(test_input) if isinstance(test_input, str) else test_input
                expected = eval(test_output) if isinstance(test_output, str) else test_output
            try:
                if isinstance(arg, list):
                    result = func(*arg)
                else:
                    result = func(arg)
            except TypeError:
                result = func(arg)
            if result != expected:
                raise AssertionError(f'Input: {arg}, Expected: {expected}, Got: {result}')
        result_queue.put('passed')
    except Exception as e:
        result_queue.put(f'failed: {type(e).__name__}: {str(e)}')


def evaluate_lcb(sample, completion_code):
    inputs = sample['public_test_cases']['input']
    outputs = sample['public_test_cases']['output']
    entry_point = 'solve'
    starter_code = sample.get('starter_code', '')
    if starter_code:
        try:
            tree = ast.parse(starter_code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    entry_point = node.name
                    break
                elif isinstance(node, ast.ClassDef):
                    entry_point = node.name
        except:
            pass
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=run_test_case_lcb,
        args=(completion_code, inputs, outputs, entry_point, result_queue)
    )
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
        return False, 'Process Crash'


@scope
def run_benchmark(config, num_samples=None):
    load_dotenv()
    setup_directories()
    logger.info('🤖 GCRI Worker Initializing for LiveCodeBench (Hard Mode)...')
    worker = GCRI(config, schema=LCBSolution)
    logger.info('📚 Loading LiveCodeBench dataset...')
    dataset = load_dataset('livecodebench/lcb_v1', split='test')
    dataset = dataset.filter(lambda x: x['difficulty'] in ['medium', 'hard'])
    logger.info(f'📉 Filtered (Medium/Hard) size: {len(dataset)}')
    if num_samples:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
    results = []
    total_passed = 0
    total_processed = 0
    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='Benchmarking'):
        task_id = f'LCB/{item['question_id']}'
        try:
            question_content = item['question_content']
            starter_code = item.get('starter_code', '')
            task_prompt = (
                f'You are a competitive programmer.\n'
                f'Solve the following algorithmic problem efficiently.\n'
                f'Use the provided function signature/class structure.\n\n'
                f'Problem Description:\n{question_content}\n\n'
                f'Starter Code:\n{starter_code}\n\n'
                f'Provide the reasoning and the complete, working Python code.'
            )
            logger.info(f'▶ Running Task: {task_id} ({item['difficulty']})')
            output_state = worker(task_prompt, auto_commit=True)
            final_output_obj = output_state.get('final_output')
            parsed_code = ''
            parsed_reasoning = ''
            if final_output_obj:
                if isinstance(final_output_obj, dict):
                    parsed_code = preprocess_code(final_output_obj.get('solution_code', ''))
                    parsed_reasoning = final_output_obj.get('thought_process', '')
                else:
                    parsed_code = preprocess_code(str(final_output_obj))
            is_passed, msg = evaluate_lcb(item, parsed_code)
            total_processed += 1
            if is_passed:
                total_passed += 1
            logger.info(
                f'🧪 {task_id}: {'✅ PASS' if is_passed else '❌ FAIL'} ({msg}) | '
                f'Acc: {(total_passed/total_processed)*100:.1f}%'
            )
            results.append(
                {
                    'task_id': task_id,
                    'difficulty': item['difficulty'],
                    'passed': is_passed,
                    'error': msg,
                    'code': parsed_code,
                    'reasoning': parsed_reasoning
                }
            )
            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            logger.error(f'Error on {task_id}: {e}')
            continue


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_benchmark()
