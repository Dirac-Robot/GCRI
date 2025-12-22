import json
import os
import re

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from gcri.graphs.gcri_unit import GCRI
from gcri.config import scope

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
BENCHMARK_DIR = 'benchmark_results/simplebench'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'simplebench_results_with_score.json')
DATASET_PATH = "Impulse2000/simple_bench_public-20-12-2024"


# -------------------------------------------------------------------------
# Schema Definition
# -------------------------------------------------------------------------
class SimpleBenchResult(BaseModel):
    thought_process: str = Field(
        ...,
        description='Detailed step-by-step reasoning to arrive at the correct option.'
    )
    final_answer: str = Field(
        ...,
        description='The single letter corresponding to the correct option (e.g., "A", "B", "C", "D", "E", or "F").'
    )


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


def normalize_answer(answer_str: str) -> str:
    if not answer_str:
        return ""
    clean = answer_str.strip().upper()
    if len(clean) == 1 and clean in "ABCDEF":
        return clean
    match = re.search(r'\b([A-F])\b', clean)
    if match:
        return match.group(1)
    return clean


def evaluate_answer(sample, completion_answer):
    # SimpleBench nested data usually has 'answer' field
    ground_truth = str(sample.get('answer', '')).strip().upper()
    model_answer = normalize_answer(completion_answer)

    if not ground_truth:
        return False, "Error: No ground truth in dataset"

    if model_answer == ground_truth:
        return True, "Passed"
    else:
        return False, f"Wrong Answer (Expected: {ground_truth}, Got: {model_answer})"


# -------------------------------------------------------------------------
# Main Benchmark Loop
# -------------------------------------------------------------------------
@scope
def run_benchmark(config, num_samples=None):
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()

    logger.info('🤖 GCRI Worker Initializing for SimpleBench (Reasoning Mode)...')
    worker = GCRI(config, schema=SimpleBenchResult)

    logger.info(f'📚 Loading SimpleBench dataset from {DATASET_PATH}...')
    try:
        raw_dataset = load_dataset(DATASET_PATH, split='train')
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

    # ---------------------------------------------------------------------
    # 🛠️ DATASET FLATTENING LOGIC (Fix for nested 'eval_data')
    # ---------------------------------------------------------------------
    dataset_items = []

    # Check if the dataset is wrapped in 'eval_data'
    if 'eval_data' in raw_dataset.column_names:
        logger.info("📦 Detected nested 'eval_data' structure. Flattening...")
        # Usually it's in the first row if it's a single entry dataset
        for row in raw_dataset:
            if isinstance(row.get('eval_data'), list):
                dataset_items.extend(row['eval_data'])
    else:
        # Fallback to standard iteration if structure changes
        logger.info("📦 Detected standard dataset structure.")
        dataset_items = [item for item in raw_dataset]

    logger.info(f"📊 Total items found: {len(dataset_items)}")

    if num_samples:
        dataset_items = dataset_items[:num_samples]
        logger.info(f'🔍 Running on first {num_samples} samples.')

    # ---------------------------------------------------------------------
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
                    if comp:
                        valid_results.append(item)
                        processed_ids.add(str(t_id))  # Ensure string comparison
                results = valid_results
                total_processed = len(results)
                total_passed = sum(1 for item in results if item.get('passed', False))
                logger.info(f'🔄 Resuming... {total_processed} valid items retained.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    # Iterate over the flattened list
    for idx, item in tqdm(enumerate(dataset_items), total=len(dataset_items), desc='Benchmarking'):
        # Robust ID extraction
        task_id = str(item.get('id', item.get('question_id', idx)))

        if task_id in processed_ids:
            continue

        try:
            prompt_text = item.get('prompt', '')

            task_prompt = (
                f'You are an expert at logical reasoning and solving complex trick questions.\n'
                f'Analyze the following multiple-choice question carefully.\n\n'
                f'Question:\n{prompt_text}\n\n'
                f'Think step-by-step to avoid common pitfalls or intuitive errors.\n'
                f'Select the single best answer from the options (A, B, C, D, E, or F).'
            )

            logger.info(f'▶ Running Task: {task_id}')

            output_state = worker(task_prompt, commit_mode='auto-reject')
            final_output_obj = output_state.get('final_output')

            parsed_answer = ''
            parsed_reasoning = ''

            if final_output_obj:
                if isinstance(final_output_obj, dict):
                    raw_answer = final_output_obj.get('final_answer', '')
                    parsed_answer = normalize_answer(raw_answer)
                    parsed_reasoning = final_output_obj.get('thought_process', '')
                    raw_dump = final_output_obj
                else:
                    raw_dump = str(final_output_obj)
                    parsed_answer = normalize_answer(str(final_output_obj))
            else:
                raw_dump = 'No final output generated.'

            is_passed, eval_message = evaluate_answer(item, parsed_answer)

            total_processed += 1
            if is_passed:
                total_passed += 1

            current_accuracy = (total_passed/total_processed)*100

            status_icon = '✅ PASS' if is_passed else '❌ FAIL'
            logger.info(
                f'🧪 {status_icon} | Ans: {parsed_answer} (Ref: {item.get("answer")}) | Acc: {current_accuracy:.2f}%'
            )

            result = {
                'task_id': task_id,
                'prompt': prompt_text,
                'ground_truth': item.get('answer'),
                'completion': parsed_answer,
                'reasoning': parsed_reasoning,
                'passed': is_passed,
                'eval_message': eval_message,
                'raw_output': raw_dump,
                'full_state': {
                    'best_branch': output_state.get('best_branch_index'),
                    'decision': output_state.get('decision'),
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

    final_acc = (total_passed/len(dataset_items))*100 if len(dataset_items) > 0 else 0
    logger.info(f'✅ Benchmark completed. Final Accuracy: {final_acc:.2f}%')
    logger.info(f'📄 Detailed results saved to {RESULT_FILE}')


if __name__ == '__main__':
    run_benchmark()