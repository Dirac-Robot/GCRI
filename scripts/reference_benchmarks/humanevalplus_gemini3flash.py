import json
import multiprocessing
import os

from datasets import load_dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from tqdm import tqdm

DATASET_HF_ID = "evalplus/humanevalplus"
BENCHMARK_DIR = 'benchmark_results/humanevalplus'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'humanevalplus_results_gemini3flash.json')
MODEL_NAME = 'gemini-3-flash-preview'
TIMEOUT_SECONDS = 5.


def setup():
    load_dotenv()
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
    return code_str.strip()


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

    entry_point = sample['entry_point']

    plus_data = sample.get('plus', {})
    plus_inputs = plus_data.get('input', [])
    plus_outputs = plus_data.get('output', [])
    atol = plus_data.get('atol', 0)

    full_code = header+completion_code+'\n\n'

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


def generate_code(llm, function_prompt: str) -> str:
    task_prompt = (
        f'You are an expert Python software engineer.\n'
        f'Implement the following Python function completely.\n'
        f'Return the COMPLETE function including the signature, docstring, and implementation.\n\n'
        f'--- FUNCTION TO IMPLEMENT ---\n'
        f'{function_prompt}\n\n'
        f'Provide ONLY the complete function code without any explanations.'
    )

    try:
        response = llm.invoke(task_prompt)
        content = response.content if hasattr(response, 'content') else response

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and 'text' in part:
                    text_parts.append(part['text'])
                elif isinstance(part, str):
                    text_parts.append(part)
            response_text = '\n'.join(text_parts)
        elif isinstance(content, dict) and 'text' in content:
            response_text = content['text']
        else:
            response_text = str(content)

        return preprocess_code(response_text)
    except Exception as e:
        logger.error(f'Generation error: {e}')
        return ''


def run_benchmark(num_samples=None):
    setup()

    logger.info(f'🤖 Initializing {MODEL_NAME}...')
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2, thinking_level='medium')

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
                results = valid_results
                total_processed = len(results)
                total_passed = sum(1 for item in results if item.get('passed', False))
                logger.info(f'🔄 Resuming... {total_processed} valid items retained.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc=f'HumanEval+ ({MODEL_NAME})'):
        task_id = item.get('task_id')
        if task_id in processed_ids:
            continue

        try:
            function_prompt = item.get('prompt', '')

            logger.info(f'▶ Running Task: {task_id}')
            completion = generate_code(llm, function_prompt)
            logger.info(f'   Generated ({len(completion)} chars): {repr(completion[:100])}...')

            is_passed, eval_message = evaluate_code(item, completion)

            total_processed += 1
            if is_passed:
                total_passed += 1

            current_accuracy = (total_passed/total_processed)*100
            logger.info(
                f'🧪 Result: {"✅ PASS" if is_passed else "❌ FAIL"} ({eval_message}) | Acc: {current_accuracy:.2f}%'
            )

            result = {
                'task_id': task_id,
                'prompt': function_prompt,
                'completion': completion,
                'passed': is_passed,
                'error_log': eval_message,
                'model': MODEL_NAME
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
