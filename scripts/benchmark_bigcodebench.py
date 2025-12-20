import json
import multiprocessing
import os
import sys

# BigCodeBench requires these libraries usually
import pandas as pd
import numpy as np

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from gcri.graphs.gcri_unit import GCRI
from gcri.config import scope

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BENCHMARK_DIR = 'benchmark_results/bigcodebench'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'bigcodebench_hard_results.json')
TIMEOUT_SECONDS = 10.0  # Libraries take longer to load/run than pure Python


class BigCodeBenchResult(BaseModel):
    thought_process: str = Field(
        ...,
        description='Detailed reasoning about which libraries to use and how to implement the solution.'
    )
    solution_code: str = Field(
        ...,
        description='The complete, executable Python code implementation. Must include necessary imports inside.'
    )


def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


def preprocess_code(code_str: str) -> str:
    code_str = code_str.strip()
    if code_str.startswith("```python"):
        code_str = code_str[9:]
    elif code_str.startswith("```"):
        code_str = code_str[3:]
    if code_str.endswith("```"):
        code_str = code_str[:-3]
    return code_str.strip()


def run_test_case(full_code, result_queue):
    """
    Executes the code in a separate process.
    Note: In a real production eval, this should be sandboxed (Docker).
    For local testing, we assume the host has necessary libs (pandas, numpy, etc.)
    """
    try:
        # Prepare execution environment with common libs pre-imported just in case,
        # though the model should import what it needs.
        exec_globals = {
            'pd': pd,
            'np': np,
        }
        # BigCodeBench tests often require specific setup; execution capture is tricky.
        # We run the prompt + solution + test code.
        exec(full_code, exec_globals)
        result_queue.put("passed")
    except AssertionError:
        result_queue.put("failed: Assertion Error")
    except Exception as e:
        result_queue.put(f"failed: {str(e)}")


def evaluate_code(sample, completion_code):
    # BigCodeBench uses specific imports in its tests.
    # We prepend common imports to ensure execution context has them if the test snippet assumes them.
    header = (
        'import math\n'
        'import string\n'
        'import re\n'
        'import collections\n'
        'import heapq\n'
        'import itertools\n'
        'import functools\n'
        'import sys\n'
        'import pandas as pd\n'
        'import numpy as np\n'
        'from typing import *\n\n'
    )

    # BigCodeBench structure:
    # complete_prompt: The prompt given to the model (imports + docstring)
    # test: The unit test code (asserts)
    # entry_point: The function name
    # We need to be careful: completion_code should merge with the prompt correctly.
    # Often 'complete_prompt' has the signature.
    full_code = (
        header+
        completion_code+'\n\n'+
        sample['test']+'\n\n'  # The test code typically calls the function and asserts
    )

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
        # If queue is empty but no timeout, it likely crashed silently or finished without writing (rare)
        return False, 'Process Crash/No Result'


@scope
def run_benchmark(config, num_samples=None):
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()

    logger.info('🤖 GCRI Worker Initializing for BigCodeBench (Tool Use Mode)...')
    worker = GCRI(config, schema=BigCodeBenchResult)

    logger.info('📚 Loading BigCodeBench dataset (Hard subset)...')
    try:
        # Load the specific version used for evaluation
        dataset = load_dataset('bigcode/bigcodebench', split='v0.1.2')
        # Filter for 'Hard' tasks if possible, or run all.
        # BigCodeBench implies hard by nature, but let's filter specifically if metadata exists.
        # For simplicity in this script, we run the dataset as is, which is already tough.
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

    # Optional: Filter only hard ones if metadata allows, or just slice.
    # We will just take the first N samples as requested.
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
                    if t_id:
                        valid_results.append(item)
                        processed_ids.add(t_id)
                results = valid_results
                total_processed = len(results)
                total_passed = sum(1 for item in results if item.get('passed', False))
                logger.info(f'🔄 Resuming... {total_processed} items retained.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='Benchmarking'):
        task_id = item.get('task_id')
        if task_id in processed_ids:
            continue

        try:
            # complete_prompt includes imports and function signature
            prompt_content = item.get('complete_prompt', '')

            task_prompt = (
                f"You are an expert Python Data Scientist and Engineer.\n"
                f"Complete the following Python function. You may need to use libraries like pandas, numpy, etc.\n"
                f"The environment has standard data science libraries installed.\n\n"
                f"{prompt_content}\n\n"
                f"Provide the reasoning and the fully functional code implementation."
            )

            logger.info(f'▶ Running Task: {task_id}')
            output_state = worker(task_prompt, auto_commit=True)
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

            # Execute Check
            # Note: BigCodeBench prompts often already have imports. We append the model's body.
            # But the model might rewrite imports. The 'complete_prompt' is what should be 'seen',
            # but the execution needs the valid full code.
            # Strategy: Combine model code. If model repeats imports, Python handles it fine.
            # Combine logic: Sometimes prompts are just signatures.
            # We trust the model to output the 'def ...' part or the body?
            # Usually HumanEval style models output the function body or whole function.
            # We assume the model outputs the WHOLE function based on the prompt.
            is_passed, eval_message = evaluate_code(item, parsed_code)

            total_processed += 1
            if is_passed:
                total_passed += 1

            current_accuracy = (total_passed/total_processed)*100

            logger.info(
                f'🧪 Result: {"✅ PASS" if is_passed else "❌ FAIL"} ({eval_message}) | Acc: {current_accuracy:.2f}%'
            )

            result = {
                'task_id': task_id,
                'prompt': prompt_content,
                'completion': parsed_code,
                'reasoning': parsed_reasoning,
                'passed': is_passed,
                'error_log': eval_message,
                'full_state': {
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
    logger.info(f'✅ Benchmark completed. Final Accuracy: {final_acc:.2f}%')
    logger.info(f'📄 Detailed results saved to {RESULT_FILE}')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_benchmark()