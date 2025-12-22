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

# --- Configuration ---
BENCHMARK_DIR = 'benchmark_results/proofwriter'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'proofwriter_results_depth5plus.json')
DATASET_CONFIG = "open_world_assumption"  # OWA is harder and includes 'Unknown'
TARGET_DEPTH_MIN = 5  # Filter for depth >= 5


class ProofWriterResult(BaseModel):
    chain_of_thought: str = Field(
        ...,
        description="The step-by-step logical deduction proof."
    )
    final_label: str = Field(
        ...,
        description="The final conclusion. Must be one of: 'True', 'False', or 'Unknown'."
    )


def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


# --- Robust Evaluation Logic (The "Anti-Oga" Layer) ---

def normalize_label(pred_text: str) -> str:
    """
    Maps various model outputs to standard ProofWriter labels: True, False, Unknown.
    Handles synonyms and messy outputs.
    """
    if not isinstance(pred_text, str):
        return "error"

    text = pred_text.strip().lower()

    # 1. Clear "Unknown" patterns (Priority because usually phrased uniquely)
    unknown_patterns = [
        'unknown', 'uncertain', 'cannot be determined', 'cannot determine',
        'not enough information', 'insufficient information', 'undetermined',
        'neither true nor false'
    ]
    if any(p in text for p in unknown_patterns):
        return "Unknown"

    # 2. Clear "True" patterns
    true_patterns = ['true', 'yes', 'correct', 'proved', 'is true']
    # Check strict equality first for short answers
    if text in true_patterns:
        return "True"
    # Check presence for longer answers, but be careful of negation
    for p in true_patterns:
        if p in text and "not" not in text:
            return "True"

    # 3. Clear "False" patterns
    false_patterns = ['false', 'no', 'incorrect', 'disproved', 'is false', 'not true']
    if any(p in text for p in false_patterns):
        return "False"

    # 4. Fallback: try to find the last strong keyword
    tokens = re.findall(r"\b(true|false|unknown)\b", text)
    if tokens:
        return tokens[-1].capitalize()  # Return the last found logical token

    return "Error"


def evaluate_proof(pred_raw, gt_raw):
    """
    Compares prediction against Ground Truth label.
    """
    pred_norm = normalize_label(str(pred_raw))
    gt_norm = str(gt_raw).capitalize()  # ProofWriter GT is usually "True"/"False"/"Unknown"

    if pred_norm == gt_norm:
        return True, f"Match ({pred_norm})"

    return False, f"Mismatch (Pred: '{pred_raw}' -> {pred_norm}, GT: {gt_norm})"


def format_input_prompt(item):
    """
    Formats the context and question clearly.
    """
    theory = item.get('theory', '')
    questions = item.get('questions', {})

    # ProofWriter dataset structure on HF can be tricky.
    # Usually 'theory' is a string and 'questions' is a list/dict.
    # We will process one question at a time in the loop.
    return theory  # This is just the context part


# --- Main Benchmark Loop ---

@scope
def run_benchmark(config, num_samples=None):
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()

    logger.info('🤖 GCRI Worker Initializing for ProofWriter (Deep Logic Mode)...')
    worker = GCRI(config, schema=ProofWriterResult)

    logger.info(f'📚 Loading dataset: allenai/proofwriter ({DATASET_CONFIG})...')
    try:
        # Load the test set.
        # Note: ProofWriter structure has 'theory', 'questions' (list).
        # We need to flatten this to benchmark individual questions.
        dataset = load_dataset('allenai/proofwriter', DATASET_CONFIG, split='test')
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

    # --- Flattening & Filtering for Depth ---
    # ProofWriter items contain multiple questions per theory.
    # We need to extract questions with depth >= TARGET_DEPTH_MIN.
    logger.info(f'🔍 Filtering samples for Depth >= {TARGET_DEPTH_MIN}...')
    benchmark_tasks = []

    # Pre-scan dataset to flatten valid tasks
    for item in tqdm(dataset, desc="Preparing Tasks"):
        theory = item.get('theory', '')
        questions = item.get('questions', [])

        for q in questions:
            q_depth = q.get('QDepth', 0)
            if q_depth >= TARGET_DEPTH_MIN:
                benchmark_tasks.append(
                    {
                        'id': q.get('id'),  # Unique ID
                        'theory': theory,
                        'question_text': q.get('question', ''),
                        'answer': str(q.get('answer', '')),  # True/False/Unknown
                        'depth': q_depth
                    }
                )

    logger.info(f'✅ Found {len(benchmark_tasks)} tasks with Depth >= {TARGET_DEPTH_MIN}.')

    if num_samples:
        benchmark_tasks = benchmark_tasks[:num_samples]
        logger.info(f'✂️ Limiting to first {num_samples} samples.')

    # --- Resumption Logic ---
    results = []
    processed_ids = set()
    total_processed = 0
    total_passed = 0

    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for item in existing_data:
                    t_id = item.get('task_id')
                    if item.get('completion'):
                        results.append(item)
                        processed_ids.add(t_id)
                total_processed = len(results)
                total_passed = sum(1 for item in results if item.get('passed', False))
                logger.info(f'🔄 Resuming... {total_processed} items loaded.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    # --- Processing ---
    for task in tqdm(benchmark_tasks, desc='Benchmarking'):
        task_id = task['id']
        if task_id in processed_ids:
            continue

        try:
            # Context construction
            theory_text = task['theory']
            question_text = task['question_text']
            ground_truth = task['answer']
            depth = task['depth']

            task_prompt = (
                f"You are a logic engine capable of multi-hop deductive reasoning.\n"
                f"Analyze the following facts and rules (Theory) and determine if the Statement is True, False, or Unknown.\n"
                f"Use the Open World Assumption (if something is not stated or cannot be derived, it is Unknown).\n\n"
                f"=== Theory ===\n{theory_text}\n\n"
                f"=== Statement ===\n{question_text}\n\n"
                f"Output Instruction:\n"
                f"1. In 'chain_of_thought', write out the derivation steps clearly.\n"
                f"2. In 'final_label', output ONLY one of: 'True', 'False', or 'Unknown'."
            )

            # Execution
            output_state = worker(task_prompt, commit_mode='auto-reject')
            final_output_obj = output_state.get('final_output')

            parsed_label = ''
            parsed_reasoning = ''
            raw_dump = ''

            if final_output_obj and isinstance(final_output_obj, dict):
                parsed_label = str(final_output_obj.get('final_label', '')).strip()
                parsed_reasoning = final_output_obj.get('chain_of_thought', '')
                raw_dump = final_output_obj
            else:
                # Fallback for plain text response
                raw_dump = str(final_output_obj)
                parsed_label = str(final_output_obj)  # Will be normalized later

            # --- Evaluation ---
            is_passed, eval_message = evaluate_proof(parsed_label, ground_truth)

            total_processed += 1
            if is_passed:
                total_passed += 1

            current_accuracy = (total_passed/total_processed)*100

            if not is_passed:
                # Log failures to debug "Oga" or model capability
                logger.warning(f"❌ FAIL (D{depth}) | Pred: {parsed_label} | GT: {ground_truth}")

            logger.info(f"📊 Acc: {current_accuracy:.2f}% (Depth {depth} task)")

            result_entry = {
                'task_id': task_id,
                'depth': depth,
                'question': question_text,
                'theory': theory_text,
                'ground_truth': ground_truth,
                'completion': parsed_label,
                'chain_of_thought': parsed_reasoning,
                'passed': is_passed,
                'eval_message': eval_message,
                'raw_output': raw_dump
            }
            results.append(result_entry)

            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except KeyboardInterrupt:
            logger.warning('⛔ Benchmark interrupted by user.')
            break
        except Exception as e:
            logger.error(f'❌ Error processing task {task_id}: {e}')
            continue

    final_acc = (total_passed/len(benchmark_tasks))*100 if len(benchmark_tasks) > 0 else 0
    logger.info(f'✅ Benchmark completed. Final Accuracy (Depth>={TARGET_DEPTH_MIN}): {final_acc:.2f}%')


if __name__ == '__main__':
    run_benchmark()