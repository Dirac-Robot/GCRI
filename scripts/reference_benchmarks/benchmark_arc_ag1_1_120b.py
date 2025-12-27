import json
import os
import re
import ast
import glob
import numpy as np
import time
import shutil
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "openai/gpt-oss-120b"
API_KEY = "EMPTY"
MAX_TOKENS = 4096
TEMPERATURE = 0.0
USE_STRUCTURED_OUTPUT = False

DATASET_HF_ID = "lordspline/arc-agi"
LOCAL_DATA_PATH = "data/arc"
TARGET_SPLIT = "evaluation"
BENCHMARK_DIR = 'benchmark_results/arc_agi_single'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'arc_results_single_agent.json')

def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

def format_grid(grid):
    return str(grid).replace('],', '],\n')

def extract_grid_from_text(text: str):
    if not text: return None
    text = str(text).strip()
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```python\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    pattern = r"\[\s*\[.*?\]\s*\]"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        candidate = matches[-1]
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except:
            pass
    return None

def check_answer(pred_grid, gt_grid):
    if pred_grid is None:
        return False, "Parse Error (No Grid Found)"
    try:
        pred_arr = np.array(pred_grid)
        gt_arr = np.array(gt_grid)
        if pred_arr.ndim != gt_arr.ndim: return False, f"Dimension Mismatch"
        if pred_arr.shape != gt_arr.shape:
            return False, f"Shape Mismatch (Pred: {pred_arr.shape}, GT: {gt_arr.shape})"
        if np.array_equal(pred_arr, gt_arr):
            return True, "Exact Match"
        else:
            return False, f"Value Mismatch"
    except Exception as e:
        return False, f"Comparison Error: {str(e)}"

def load_arc_data(split_name, num_samples=None):
    dataset_items = []
    try:
        logger.info(f"☁️ Loading HF: {DATASET_HF_ID} [{split_name}]")
        ds = load_dataset(DATASET_HF_ID, split=split_name)
        if num_samples:
            ds = ds.select(range(min(len(ds), num_samples)))

        for idx, item in enumerate(ds):
            original_id = item.get('id')
            if original_id is None or str(original_id).strip() == '':
                task_id = f"task_{idx:03d}"
            else:
                task_id = str(original_id)

            dataset_items.append({
                'id': task_id,
                'train': item['train'],
                'test': item['test']
            })
        return dataset_items
    except Exception as e:
        logger.warning(f"⚠️ HF Load Failed: {e}")

    local_dir = os.path.join(LOCAL_DATA_PATH, split_name)
    if os.path.exists(local_dir):
        files = glob.glob(os.path.join(local_dir, '*.json'))
        files.sort()
        if num_samples:
            files = files[:num_samples]
        for fpath in files:
            with open(fpath, 'r') as f:
                data = json.load(f)
                dataset_items.append({
                    'id': os.path.basename(fpath).replace('.json', ''),
                    'train': data['train'],
                    'test': data['test']
                })
        return dataset_items
    return []

def run_benchmark(num_samples=None):
    load_dotenv()
    setup_directories()

    # 결과 파일 초기화 여부 확인 (이전의 corrupted 파일 방지)
    if os.path.exists(RESULT_FILE):
        logger.warning(f"⚠️ 기존 결과 파일 발견: {RESULT_FILE}")

    logger.info(f'🤖 Initializing Agent (Model: {MODEL_NAME})')

    llm = ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=API_KEY,
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        timeout=300,
        max_retries=2,
    )

    dataset = load_arc_data(TARGET_SPLIT, num_samples)
    if not dataset:
        logger.error("No data found.")
        return

    results = []
    processed_ids = set()

    # 통계 변수 초기화
    total_processed = 0
    total_passed = 0

    # Resume Logic
    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                for item in existing:
                    if item['task_id'] == 'unknown': continue # 잘못된 ID 스킵

                    if item.get('completion') or item.get('passed') is False:
                        results.append(item)
                        processed_ids.add(item['task_id'])

                        # Resume 시 통계 복구
                        total_processed += 1
                        if item.get('passed', False):
                            total_passed += 1

            logger.info(f'🔄 Resuming... {total_processed} items loaded. (Curr Acc: {(total_passed/total_processed*100):.2f}%)')
        except:
            logger.warning('⚠️ Result file corrupt, starting fresh.')

    for item in tqdm(dataset, desc='Benchmarking'):
        task_id = item['id']
        if task_id in processed_ids:
            continue

        train_pairs = item['train']
        test_pairs = item['test']

        # 프롬프트 구성 생략 (이전과 동일)
        example_str = ""
        for i, pair in enumerate(train_pairs):
            example_str += (
                f"Example {i+1}:\n"
                f"Input:\n{format_grid(pair['input'])}\n"
                f"Output:\n{format_grid(pair['output'])}\n\n"
            )

        task_passed = True
        task_logs = []

        # 테스트 케이스 실행
        for t_idx, test_pair in enumerate(test_pairs):
            test_input = test_pair['input']
            gt = test_pair['output']

            system_msg = (
                "You are an expert at the ARC (Abstraction and Reasoning Corpus) challenge.\n"
                "Determine the pattern from the examples and solve the test input.\n"
                "Output Format:\n"
                "First, explain the logic in natural language.\n"
                "Then, provide the final grid inside a Python list format like [[1,0],[0,1]]."
            )

            user_msg = (
                f"{example_str}"
                f"--- TEST INPUT ---\n"
                f"Input:\n{format_grid(test_input)}\n\n"
                f"Analyze the pattern and provide the Solution Grid."
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                ("user", user_msg),
            ])

            start_time = time.time()
            logger.info(f"⏳ [Task {task_id}] Invoking LLM...")

            try:
                chain = prompt | llm
                response = chain.invoke({})
                raw_content = response.content

                elapsed = time.time() - start_time
                parsed_grid = extract_grid_from_text(raw_content)
                is_correct, msg = check_answer(parsed_grid, gt)

                if not is_correct:
                    task_passed = False

                # 상세 로그 대신 아래에서 통합 로그 출력
                # logger.debug(f"   Test {t_idx} Result: {msg}")

                task_logs.append({
                    'test_index': t_idx,
                    'passed': is_correct,
                    'msg': msg,
                    'pred': parsed_grid,
                    'gt': gt,
                    'raw': raw_content
                })

            except Exception as e:
                logger.error(f"❌ Error in Task {task_id}: {e}")
                task_passed = False
                task_logs.append({'error': str(e)})

        # --- 통계 업데이트 및 실시간 로그 출력 ---
        total_processed += 1
        if task_passed:
            total_passed += 1

        current_acc = (total_passed / total_processed) * 100
        result_msg = "✅ PASS" if task_passed else "❌ FAIL"

        # [핵심 변경] 여기에 Acc 표시 추가
        logger.info(f"Task {task_id}: {result_msg} | 📈 Acc: {current_acc:.2f}% ({total_passed}/{total_processed})")

        result_entry = {
            'task_id': task_id,
            'passed': task_passed,
            'logs': task_logs
        }
        results.append(result_entry)
        processed_ids.add(task_id)

        try:
            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            logger.error(f"⚠️ Failed to save results: {e}")

    final_acc = (total_passed / len(dataset)) * 100 if dataset else 0
    logger.info(f'✅ Completed. Final Accuracy: {final_acc:.2f}% ({total_passed}/{len(dataset)})')

if __name__ == '__main__':
    # 기존에 생성된 잘못된 결과 파일 삭제 (권장)
    if os.path.exists(RESULT_FILE):
        try:
            # 안전을 위해 사용자에게 묻지 않고 바로 삭제하지는 않지만,
            # 'unknown' 이슈가 있었다면 수동 삭제 후 실행하는 것이 좋습니다.
            pass
        except:
            pass

    run_benchmark()