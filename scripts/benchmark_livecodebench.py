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
    # 1. public_test_cases 파싱 (문자열 -> 딕셔너리 변환)
    raw_cases = sample.get('public_test_cases', {})
    test_cases = {}

    if isinstance(raw_cases, dict):
        test_cases = raw_cases
    elif isinstance(raw_cases, str):
        # 1차 시도: JSON 표준 파싱 (Double Quotes)
        try:
            test_cases = json.loads(raw_cases)
            # 이중 인코딩 처리: 파싱 결과가 여전히 문자열이면 한 번 더 시도
            while isinstance(test_cases, str):
                test_cases = json.loads(test_cases)
        except json.JSONDecodeError:
            # 2차 시도: Python 리터럴 파싱 (Single Quotes 등) - LCB 데이터셋은 이 경우가 많음
            try:
                test_cases = ast.literal_eval(raw_cases)
                # 이중 인코딩 처리
                while isinstance(test_cases, str):
                    try:
                        test_cases = json.loads(test_cases)
                    except:
                        test_cases = ast.literal_eval(test_cases)
            except Exception as e:
                return False, f"Test case parsing failed: {str(e)}"
    else:
        return False, f"Unknown test case format: {type(raw_cases)}"

    # 리스트 형태 처리: [{"input": x, "output": y}, ...] or [[inputs...], [outputs...]]
    if isinstance(test_cases, list):
        if len(test_cases) > 0:
            # Case 1: 리스트의 각 요소가 딕셔너리 (각 테스트 케이스가 dict)
            if isinstance(test_cases[0], dict) and 'input' in test_cases[0] and 'output' in test_cases[0]:
                inputs = [tc['input'] for tc in test_cases]
                outputs = [tc['output'] for tc in test_cases]
            # Case 2: [[inputs...], [outputs...]] 형태
            elif len(test_cases) == 2 and isinstance(test_cases[0], list) and isinstance(test_cases[1], list):
                inputs = test_cases[0]
                outputs = test_cases[1]
            else:
                return False, f"Unsupported list format: {test_cases[:2] if len(test_cases) > 1 else test_cases}"
        else:
            return False, "Empty test cases list"
    # 딕셔너리 형태 처리: {"input": [...], "output": [...]}
    elif isinstance(test_cases, dict):
        try:
            inputs = test_cases['input']
            outputs = test_cases['output']
        except KeyError:
            return False, "Missing 'input' or 'output' fields in test cases"
    else:
        return False, f"Test cases parsed to {type(test_cases).__name__}, unsupported type"

    # 3. 엔트리 포인트 탐색
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

    # 4. 멀티프로세싱 실행
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
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()
    logger.info('🤖 GCRI Worker Initializing for LiveCodeBench (Hard Mode)...')
    worker = GCRI(config, schema=LCBSolution)
    logger.info('📚 Loading LiveCodeBench dataset...')
    try:
        # release_v2 등 특정 버전을 명시할 수도 있습니다. (기본값은 최신)
        dataset = load_dataset(
            'livecodebench/code_generation_lite',
            split='test',
            trust_remote_code=True
        )
        logger.info("✅ Loaded 'code_generation_lite' dataset successfully.")
    except Exception as e:
        logger.warning(f"Lite version load failed ({e}), trying full version...")
        dataset = load_dataset(
            'livecodebench/code_generation',
            split='test',
            trust_remote_code=True
        )
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
            output_state = worker(task_prompt, commit_mode='auto-reject')

            # Handle None or non-dict output_state
            if output_state is None or not isinstance(output_state, dict):
                logger.warning(f'Worker returned invalid state for {task_id}: {type(output_state)}')
                output_state = {}

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
