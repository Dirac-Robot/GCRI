import json
import os
import re
import math
import ast
import time

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# --- Configuration ---
VLLM_BASE_URL = 'http://localhost:8000/v1'
MODEL_NAME = 'openai/gpt-oss-120b'
API_KEY = 'EMPTY'
MAX_TOKENS = 12800
TEMPERATURE = 0.0

DATASET_NAME = 'TIGER-Lab/TheoremQA'
BENCHMARK_DIR = 'benchmark_results/theoremqa_single'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'theoremqa_results_120b.json')


class TheoremQAResponse(BaseModel):
    reasoning: str = Field(..., description='Step-by-step reasoning to derive the answer.')
    final_answer: str = Field(..., description='The final answer only. Minimal text. (e.g., "5", "True", "[1, 2]")')


def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


# --- Answer Parsing & Evaluation ---

def parse_numeric(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip().replace(',', '')
        if '/' in value:
            try:
                nums = value.split('/')
                if len(nums) == 2:
                    return float(nums[0])/float(nums[1])
            except:
                pass
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val_lower = value.strip().lower()
        if val_lower in ['true', 'yes', 'correct']:
            return True
        if val_lower in ['false', 'no', 'incorrect', 'wrong']:
            return False
    return None


def parse_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except:
            pass
        if ',' in value:
            parts = [p.strip() for p in value.split(',')]
            try:
                return [float(p) for p in parts]
            except:
                pass
    return None


def clean_latex(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    text = text.strip()
    if text.startswith(r'\boxed{') and text.endswith('}'):
        text = text[7:-1]
    text = text.replace('$', '').replace('\\', '')
    return text.strip()


def compare_answers(pred_raw, gt_raw):
    pred_str = clean_latex(str(pred_raw))
    gt_str = clean_latex(str(gt_raw))

    # Boolean Comparison
    pred_bool = parse_bool(pred_str)
    gt_bool = parse_bool(gt_raw)
    if gt_bool is not None:
        if pred_bool == gt_bool:
            return True, 'Boolean Match'
        if pred_bool is not None and pred_bool != gt_bool:
            return False, f'Boolean Mismatch (Pred: {pred_bool}, GT: {gt_bool})'

    # Numeric Comparison
    pred_num = parse_numeric(pred_str)
    gt_num = parse_numeric(gt_raw)
    if gt_num is not None:
        if pred_num is not None:
            if math.isclose(pred_num, gt_num, rel_tol=1e-2, abs_tol=1e-4):
                return True, 'Numeric Match'
            else:
                return False, f'Numeric Mismatch (Pred: {pred_num}, GT: {gt_num})'
        numbers = re.findall(r'[-+]?\d*\.?\d+', pred_str)
        for num_str in numbers:
            try:
                p_val = float(num_str)
                if math.isclose(p_val, gt_num, rel_tol=1e-2, abs_tol=1e-4):
                    return True, 'Numeric Extraction Match'
            except:
                pass

    # List Comparison
    pred_list = parse_list(pred_str)
    gt_list = parse_list(gt_raw)
    if gt_list is not None and pred_list is not None:
        if len(pred_list) != len(gt_list):
            return False, 'List Length Mismatch'
        match_count = 0
        for p, g in zip(pred_list, gt_list):
            p_n = parse_numeric(p)
            g_n = parse_numeric(g)
            if p_n is not None and g_n is not None:
                if math.isclose(p_n, g_n, rel_tol=1e-2, abs_tol=1e-4):
                    match_count += 1
            elif str(p).strip() == str(g).strip():
                match_count += 1
        if match_count == len(gt_list):
            return True, 'List Match'
        return False, f'List Mismatch (Pred: {pred_list}, GT: {gt_list})'

    # String Match
    if pred_str.lower() == gt_str.lower():
        return True, 'String Match'

    return False, f"Failed All Checks (Pred: '{pred_str}', GT: '{gt_str}')"


def run_benchmark(num_samples=None):
    load_dotenv()
    setup_directories()

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
    structured_llm = llm.with_structured_output(TheoremQAResponse)

    logger.info(f'📚 Loading dataset: {DATASET_NAME}...')
    try:
        dataset = load_dataset(DATASET_NAME, split='test')
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

    # Resume Logic
    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                for item in existing:
                    t_id = item.get('task_id')
                    if item.get('completion'):
                        results.append(item)
                        processed_ids.add(t_id)
                        total_processed += 1
                        if item.get('passed', False):
                            total_passed += 1
            if total_processed > 0:
                logger.info(f'🔄 Resuming... {total_processed} items loaded. (Acc: {(total_passed/total_processed*100):.2f}%)')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file corrupt, starting fresh.')

    system_msg = (
        'You are an expert mathematician and scientist.\n'
        'Solve the following problem step-by-step.\n'
        'Output Instruction:\n'
        '- If the answer is a number, provide only the number without units.\n'
        '- If the answer is a list/vector, use format [a, b, c].\n'
        '- If the answer is True/False, output exactly "True" or "False".'
    )

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='Benchmarking TheoremQA'):
        task_id = item.get('id', str(idx))
        if task_id in processed_ids:
            continue

        question = item.get('Question', '')
        answer_type = item.get('Answer_type', 'unknown')
        ground_truth = item.get('Answer', '')

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=f'Question: {question}\n\nProvide your step-by-step solution and final answer.')
        ]

        try:
            start_time = time.time()
            total_processed += 1
            response = structured_llm.invoke(messages)
            elapsed = time.time() - start_time

            parsed_answer = response.final_answer if response else ''
            reasoning = response.reasoning if response else ''
            is_passed, eval_message = compare_answers(parsed_answer, ground_truth)
            if is_passed:
                total_passed += 1

            current_accuracy = (total_passed/total_processed)*100
            result_msg = '✅ PASS' if is_passed else '❌ FAIL'
            logger.info(f'Task {task_id}: {result_msg} | 📈 Acc: {current_accuracy:.2f}% ({total_passed}/{total_processed})')

            result_entry = {
                'task_id': task_id,
                'question': question,
                'answer_type': answer_type,
                'ground_truth': ground_truth,
                'completion': parsed_answer,
                'reasoning': reasoning,
                'passed': is_passed,
                'eval_message': eval_message,
                'elapsed_time': elapsed
            }
            results.append(result_entry)

            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except KeyboardInterrupt:
            logger.warning('⛔ Benchmark interrupted by user.')
            break
        except Exception as e:
            logger.error(f'❌ Error processing sample {task_id}: {e}')
            continue

    final_acc = (total_passed/len(dataset))*100 if len(dataset) > 0 else 0
    logger.info(f'✅ Completed. Final Accuracy: {final_acc:.2f}% ({total_passed}/{len(dataset)})')
    logger.info(f'📄 Results saved to {RESULT_FILE}')


if __name__ == '__main__':
    run_benchmark()
