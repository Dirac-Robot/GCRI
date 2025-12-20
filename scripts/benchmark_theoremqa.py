import json
import os
import re
import math
import ast

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from gcri.graphs.gcri_unit import GCRI
from gcri.config import scope

# --- Configuration ---
BENCHMARK_DIR = 'benchmark_results/theoremqa'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'theoremqa_results_with_score.json')
DATASET_NAME = "TIGER-Lab/TheoremQA"


class TheoremQAResult(BaseModel):
    thought_process: str = Field(
        ...,
        description="Step-by-step reasoning to derive the answer."
    )
    final_answer: str = Field(
        ...,
        description="The final answer only. Minimal text. (e.g., '5', 'True', '[1, 2]', 'sin(x)')"
    )


def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


# --- Robust Evaluation Logic (The "Anti-Oga" Layer) ---

def parse_numeric(value):
    """Attempt to parse a string into a float, handling fractions like '1/3'."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip().replace(',', '')  # Remove thousand separators
        # Handle fractions explicitly
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
    """Robust parsing for boolean values from text."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val_lower = value.strip().lower()
        # Direct matches
        if val_lower in ['true', 'yes', 'correct']:
            return True
        if val_lower in ['false', 'no', 'incorrect', 'wrong']:
            return False
    return None


def parse_list(value):
    """Attempt to parse string representation of list/tuple."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        # Remove LaTeX braces for vectors like \begin{pmatrix}...\end{pmatrix} if necessary
        # Simple heuristic for [a, b] or (a, b)
        try:
            # Try ast.literal_eval for standard python structures
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except:
            pass
        # Fallback: split by comma if it looks like a sequence
        if ',' in value:
            parts = [p.strip() for p in value.split(',')]
            # Try to convert all parts to numbers
            try:
                return [float(p) for p in parts]
            except:
                pass
    return None


def clean_latex(text: str) -> str:
    """Remove common LaTeX formatting like \boxed{}, $, etc."""
    if not isinstance(text, str):
        return str(text)
    text = text.strip()
    # Remove \boxed{...}
    pattern = r"\\boxed\{((?:[^{}]|(?R))*)\}"  # Recursive regex is hard in Py, use simple greedy
    # Simple un-boxing
    if text.startswith(r'\boxed{') and text.endswith('}'):
        text = text[7:-1]
    text = text.replace('$', '').replace('\\', '')
    return text.strip()


def compare_answers(pred_raw, gt_raw):
    """
    Compares prediction and ground truth robustly.
    Returns: (bool, message)
    """
    pred_str = clean_latex(str(pred_raw))
    gt_str = clean_latex(str(gt_raw))

    # 1. Boolean Comparison
    pred_bool = parse_bool(pred_str)
    gt_bool = parse_bool(gt_raw)  # GT might be actual bool type
    if gt_bool is not None:
        if pred_bool == gt_bool:
            return True, "Boolean Match"
        # If GT is bool but Pred is not valid bool, usually Fail, but check reasoning?
        if pred_bool is not None and pred_bool != gt_bool:
            return False, f"Boolean Mismatch (Pred: {pred_bool}, GT: {gt_bool})"

    # 2. Numeric Comparison (with tolerance)
    pred_num = parse_numeric(pred_str)
    gt_num = parse_numeric(gt_raw)

    if gt_num is not None:
        if pred_num is not None:
            # Tolerance: 1% relative error or 1e-4 absolute error
            if math.isclose(pred_num, gt_num, rel_tol=1e-2, abs_tol=1e-4):
                return True, "Numeric Match"
            else:
                return False, f"Numeric Mismatch (Pred: {pred_num}, GT: {gt_num})"

        # If prediction contains the number (e.g. "The answer is 5")
        # Extract all numbers and check if any match
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", pred_str)
        for num_str in numbers:
            p_val = float(num_str)
            if math.isclose(p_val, gt_num, rel_tol=1e-2, abs_tol=1e-4):
                return True, "Numeric Extraction Match"

    # 3. List/Vector Comparison
    pred_list = parse_list(pred_str)
    gt_list = parse_list(gt_raw)
    if gt_list is not None:
        if pred_list is not None:
            # Check length
            if len(pred_list) != len(gt_list):
                return False, "List Length Mismatch"
            # Compare elements (assuming order matters, or sort if needed? usually order matters in vectors)
            match_count = 0
            for p, g in zip(pred_list, gt_list):
                # Recursive numeric check for elements
                p_n = parse_numeric(p)
                g_n = parse_numeric(g)
                if p_n is not None and g_n is not None:
                    if math.isclose(p_n, g_n, rel_tol=1e-2, abs_tol=1e-4):
                        match_count += 1
                elif str(p).strip() == str(g).strip():
                    match_count += 1

            if match_count == len(gt_list):
                return True, "List Match"
            return False, f"List Mismatch (Pred: {pred_list}, GT: {gt_list})"

    # 4. Fallback: Exact String Match (Normalized)
    if pred_str.lower() == gt_str.lower():
        return True, "String Match"

    return False, f"Failed All Checks (Pred: '{pred_str}', GT: '{gt_str}')"


# --- Main Benchmark Loop ---

@scope
def run_benchmark(config, num_samples=None):
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()

    logger.info('🤖 GCRI Worker Initializing for TheoremQA...')
    worker = GCRI(config, schema=TheoremQAResult)

    logger.info(f'📚 Loading dataset: {DATASET_NAME}...')
    try:
        dataset = load_dataset(DATASET_NAME, split='test')
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

    if num_samples:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
        logger.info(f'🔍 Running on first {num_samples} samples.')

    # --- Resumption Logic ---
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
                    # TheoremQA usually uses 'id' or index. Assuming 'id' field exists.
                    t_id = item.get('task_id')
                    if item.get('completion'):  # Check basic validity
                        valid_results.append(item)
                        processed_ids.add(t_id)
                results = valid_results
                total_processed = len(results)
                total_passed = sum(1 for item in results if item.get('passed', False))
                logger.info(f'🔄 Resuming... {total_processed} items loaded.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    # --- Processing ---
    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='Benchmarking'):
        task_id = item.get('id', str(idx))  # Fallback ID
        if task_id in processed_ids:
            continue

        question = item.get('Question', '')
        answer_type = item.get('Answer_type', 'unknown')
        ground_truth = item.get('Answer', '')

        try:
            # Prompt Engineering to minimize "Oga"
            task_prompt = (
                f"You are an expert mathematician and scientist.\n"
                f"Solve the following problem carefully.\n"
                f"Question: {question}\n\n"
                f"Output Instruction:\n"
                f"- Provide step-by-step reasoning in 'thought_process'.\n"
                f"- Provide ONLY the final result in 'final_answer'.\n"
                f"- If the answer is a number, do not include units unless explicitly asked.\n"
                f"- If the answer is a list/vector, use format [a, b, c].\n"
                f"- If the answer is True/False, output 'True' or 'False'.\n"
            )

            # logger.info(f'▶ Running Task: {task_id}')
            output_state = worker(task_prompt, auto_commit=True)
            final_output_obj = output_state.get('final_output')

            parsed_answer = ''
            parsed_reasoning = ''
            raw_dump = ''

            if final_output_obj and isinstance(final_output_obj, dict):
                parsed_answer = str(final_output_obj.get('final_answer', '')).strip()
                parsed_reasoning = final_output_obj.get('thought_process', '')
                raw_dump = final_output_obj
            else:
                parsed_answer = str(final_output_obj)
                raw_dump = str(final_output_obj)

            # --- Evaluation ---
            is_passed, eval_message = compare_answers(parsed_answer, ground_truth)

            total_processed += 1
            if is_passed:
                total_passed += 1

            current_accuracy = (total_passed/total_processed)*100

            # Less verbose logging for loop, strictly result
            if not is_passed:
                logger.warning(f"❌ FAIL ID {task_id}: {eval_message}")
            # else:
            #    logger.success(f"✅ PASS ID {task_id}")
            logger.info(f"📊 Progress: {current_accuracy:.2f}% ({total_passed}/{total_processed})")

            result_entry = {
                'task_id': task_id,
                'question': question,
                'answer_type': answer_type,
                'ground_truth': ground_truth,
                'completion': parsed_answer,
                'reasoning': parsed_reasoning,
                'passed': is_passed,
                'eval_message': eval_message,
                'raw_output': raw_dump,
                'full_state': {
                    'best_branch': output_state.get('best_branch_index'),
                    'iterations': output_state.get('count', 0)
                }
            }
            results.append(result_entry)

            # Save every step
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
    run_benchmark()