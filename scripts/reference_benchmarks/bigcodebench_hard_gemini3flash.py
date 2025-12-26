import json
import multiprocessing
import os

from google import genai
from google.genai import types
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

DATASET_HF_ID = "bigcode/bigcodebench-hard"
BENCHMARK_DIR = 'benchmark_results/bigcodebench_hard'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'bigcodebench_hard_results_gemini3flash_medium.json')
TIMEOUT_SECONDS = 30.
MODEL_NAME = 'gemini-3-flash-preview'
THINKING_LEVEL = 'medium'


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
        'import hashlib\n'
        'import json\n'
        'import datetime\n'
        'import random\n'
        'import pickle\n'
        'import csv\n'
        'import io\n'
        'import tempfile\n'
        'import shutil\n'
        'import glob\n'
        'import subprocess\n'
        'import threading\n'
        'import time\n'
        'import unittest\n'
        'from typing import *\n'
        'from unittest.mock import patch, MagicMock\n\n'
    )

    test_code = sample.get('test', '')

    if not test_code:
        return False, 'No test code available'

    full_code = common_imports+completion_code+'\n\n'+test_code

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


def generate_solution(client, complete_prompt: str) -> str:
    task_prompt = (
        f'You are an expert Python software engineer.\n'
        f'Implement the following Python function completely.\n'
        f'Return the COMPLETE function including the signature, docstring, and implementation.\n'
        f'This benchmark tests complex, real-world coding with multiple library usage.\n\n'
        f'--- FUNCTION TO IMPLEMENT ---\n'
        f'{complete_prompt}\n\n'
        f'Provide ONLY the complete function code without any explanations.'
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=task_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
                temperature=0.7,
                max_output_tokens=4096,
            )
        )
        return preprocess_code(response.text)
    except Exception as e:
        logger.error(f'Generation error: {e}')
        return ''


def run_benchmark(num_samples=None, split='v0.1.4'):
    setup()

    logger.info(f'🤖 Initializing {MODEL_NAME} with thinking_level={THINKING_LEVEL}...')
    client = genai.Client()

    logger.info(f'📚 Loading BigCodeBench Hard dataset from {DATASET_HF_ID} (split: {split})...')
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
                    t_id = item.get('task_id')
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

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='BigCodeBench Hard (Gemini 3 Flash)'):
        task_id = item.get('task_id', str(idx))
        if task_id in processed_ids:
            continue

        try:
            complete_prompt = item.get('complete_prompt', item.get('code_prompt', ''))

            logger.info(f'▶ Running Task: {task_id}')
            parsed_code = generate_solution(client, complete_prompt)

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
                'prompt': complete_prompt[:1000],
                'canonical_solution': item.get('canonical_solution', '')[:500],
                'completion': parsed_code,
                'passed': is_passed,
                'error_log': eval_message,
                'model': MODEL_NAME,
                'thinking_level': THINKING_LEVEL
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
    logger.info(f'✅ BigCodeBench Hard completed. Final Accuracy: {final_acc:.2f}%')
    logger.info(f'📄 Detailed results saved to {RESULT_FILE}')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_benchmark()
