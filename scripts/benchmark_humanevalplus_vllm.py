import json
import multiprocessing
import os
import time

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
VLLM_BASE_URL = 'http://localhost:8000/v1'
MODEL_NAME = 'openai/gpt-oss-120b'
API_KEY = 'EMPTY'
MAX_TOKENS = 4096
TEMPERATURE = 0.0

DATASET_HF_ID = 'evalplus/humanevalplus'
BENCHMARK_DIR = 'benchmark_results/humanevalplus_vllm'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'humanevalplus_results_gpt_oss_120b.json')
TIMEOUT_SECONDS = 5.


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

    return '\n'.join(cleaned_lines)


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


def run_benchmark(num_samples=None):
    load_dotenv()
    setup_directories()

    logger.info(f'🤖 Initializing LLM (Model: {MODEL_NAME})')
    logger.info(f'   Base URL: {VLLM_BASE_URL}')

    llm = ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=API_KEY,
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        timeout=300,
        max_retries=2,
    )

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
                        total_processed += 1
                        if item.get('passed', False):
                            total_passed += 1
                    else:
                        logger.info(f'♻️ Re-queueing Task {t_id} (Reason: Empty completion)')
                results = valid_results
                logger.info(f'🔄 Resuming... {total_processed} valid items loaded. (Curr Acc: {(total_passed/total_processed*100):.2f}%)')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    system_msg = (
        'You are an expert Python software engineer.\n'
        'Complete the following Python function based on the provided signature and docstring.\n'
        'Your code must be valid Python and strictly follow the indentation.\n'
        'Pay special attention to edge cases as this benchmark has extensive test coverage.\n\n'
        'Output Format:\n'
        '1. First, explain your reasoning about the algorithm and edge cases.\n'
        '2. Then, provide the complete implementation code inside a ```python code block.'
    )

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='HumanEval+ Benchmark'):
        task_id = item.get('task_id')
        if task_id in processed_ids:
            continue

        try:
            function_prompt = item.get('prompt', '')
            user_msg = (
                f'Complete the following Python function:\n\n'
                f'{function_prompt}\n\n'
                f'Provide the reasoning and the fully functional implementation.'
            )

            prompt = ChatPromptTemplate.from_messages([
                ('system', system_msg),
                ('user', user_msg),
            ])

            start_time = time.time()
            logger.info(f'▶ Running Task: {task_id}')

            chain = prompt | llm
            response = chain.invoke({})
            raw_content = response.content
            elapsed = time.time() - start_time

            parsed_code = preprocess_code(raw_content)

            is_passed, eval_message = evaluate_code(item, parsed_code)

            total_processed += 1
            if is_passed:
                total_passed += 1

            current_accuracy = (total_passed/total_processed)*100
            result_msg = '✅ PASS' if is_passed else '❌ FAIL'
            logger.info(f'🧪 Result: {result_msg} ({eval_message}) | Acc: {current_accuracy:.2f}% | Time: {elapsed:.1f}s')

            result = {
                'task_id': task_id,
                'prompt': function_prompt,
                'canonical_solution': item.get('canonical_solution'),
                'completion': parsed_code,
                'passed': is_passed,
                'error_log': eval_message,
                'raw_output': raw_content,
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
    logger.info(f'✅ HumanEval+ Benchmark completed. Final Accuracy: {final_acc:.2f}% ({total_passed}/{len(dataset)})')
    logger.info(f'📄 Detailed results saved to {RESULT_FILE}')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_benchmark()
