import json
import os
import re
import ast
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from gcri.graphs.gcri_unit import GCRI
from gcri.config import scope

# --- Configuration ---

# HuggingFace Dataset: ARC-AGI-2 (1000 train, 112 eval)
DATASET_HF_ID = "eturok/ARC-AGI-2"

# Target Split: 'train' (1000 tasks) or 'test' (112 eval tasks)
TARGET_SPLIT = "test"
BENCHMARK_DIR = 'benchmark_results/arc_agi_2'


@scope
def get_preset_name(config):
    if config.get('custom_config_path'):
        return os.path.splitext(os.path.basename(config.custom_config_path))[0]
    return 'none'


RESULT_FILE = os.path.join(BENCHMARK_DIR, f'arc_results_{get_preset_name()}.json')


class ArcResult(BaseModel):
    thought_process: str = Field(
        ...,
        description="Analyze the transformation rules from train examples (colors, geometry, topology)."
    )
    solution_grid: list[list[int]] = Field(
        ...,
        description="The final output grid as a 2D integer array (e.g., [[0, 1], [2, 3]])."
    )


def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


# --- Robust Grid Extraction Logic ---

def format_grid(grid):
    """Format grid for prompt presentation."""
    return str(grid).replace('],', '],\n')


def extract_grid_from_text(text: str):
    """
    Attempts to find a 2D list pattern in text even if not formatted as JSON.
    """
    if isinstance(text, list):
        return text

    text = str(text).strip()

    # Regex for [[1,2], [3,4]] pattern spanning multiple lines
    pattern = r"\[\s*\[.*?\]\s*\]"
    matches = re.findall(pattern, text, re.DOTALL)

    candidates = []
    if not matches:
        matches = [text]

    for match in matches:
        try:
            clean_match = match.replace('```python', '').replace('```json', '').replace('```', '').strip()
            parsed = ast.literal_eval(clean_match)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                candidates.append(parsed)
        except:
            continue

    if candidates:
        return candidates[-1]
    return None


def check_answer(pred_grid, gt_grid):
    """ARC requires exact integer grid match."""
    if pred_grid is None:
        return False, "Parse Error (No Grid Found)"
    try:
        pred_arr = np.array(pred_grid)
        gt_arr = np.array(gt_grid)
        if pred_arr.shape != gt_arr.shape:
            return False, f"Shape Mismatch (Pred: {pred_arr.shape}, GT: {gt_arr.shape})"
        if np.array_equal(pred_arr, gt_arr):
            return True, "Exact Match"
        else:
            mismatch_count = np.sum(pred_arr != gt_arr)
            return False, f"Value Mismatch ({mismatch_count} pixels wrong)"
    except Exception as e:
        return False, f"Comparison Error: {str(e)}"


def load_arc_data(split_name, num_samples=None):
    """
    Load ARC-AGI-2 data from eturok/ARC-AGI-2 HuggingFace dataset.
    Dataset structure:
      - id: task identifier
      - example_inputs: list of input grids for demonstration
      - example_outputs: list of output grids for demonstration
      - question_inputs: list of input grids for test
      - question_outputs: list of output grids for test ground truth
    """
    dataset_items = []

    try:
        logger.info(f"☁️ Loading from HuggingFace: {DATASET_HF_ID} [{split_name}]")
        ds = load_dataset(DATASET_HF_ID, split=split_name)
        if num_samples:
            ds = ds.select(range(min(len(ds), num_samples)))

        for item in ds:
            train_pairs = []
            for inp, out in zip(item['example_inputs'], item['example_outputs']):
                train_pairs.append({'input': inp, 'output': out})

            test_pairs = []
            for inp, out in zip(item['question_inputs'], item['question_outputs']):
                test_pairs.append({'input': inp, 'output': out})

            dataset_items.append({
                'id': item['id'],
                'train': train_pairs,
                'test': test_pairs
            })
        logger.info(f"✅ Loaded {len(dataset_items)} tasks from {DATASET_HF_ID}")
        return dataset_items
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return []


# --- Main Benchmark Loop ---

@scope
def run_benchmark(config, num_samples=None):
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()

    logger.info('🤖 GCRI Worker Initializing for ARC-AGI...')
    worker = GCRI(config, schema=ArcResult)

    # Load Data
    dataset = load_arc_data(TARGET_SPLIT, num_samples)
    if not dataset:
        return

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
                    if item.get('completion') or item.get('passed') is False:
                        results.append(item)
                        processed_ids.add(t_id)
                total_processed = len(results)
                total_passed = sum(1 for item in results if item.get('passed', False))
                logger.info(f'🔄 Resuming... {total_processed} items loaded.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    # --- Processing ---
    for item in tqdm(dataset, desc='Benchmarking'):
        task_id = item['id']
        if task_id in processed_ids:
            continue

        train_pairs = item['train']
        test_pairs = item['test']

        # Build Context
        example_prompt = ""
        for i, pair in enumerate(train_pairs):
            example_prompt += (
                f"--- Example {i+1} ---\n"
                f"Input:\n{format_grid(pair['input'])}\n"
                f"Output:\n{format_grid(pair['output'])}\n\n"
            )

        task_passed = True
        task_logs = []

        # Evaluate Test Cases
        for t_idx, test_pair in enumerate(test_pairs):
            test_input = test_pair['input']
            test_output_gt = test_pair['output']

            task_prompt = (
                f"You are an abstract reasoning engine solving the ARC challenge.\n"
                f"Identify the transformation rule from the examples and apply it to the Test Input.\n\n"
                f"{example_prompt}"
                f"--- TEST INPUT ---\n"
                f"Input:\n{format_grid(test_input)}\n\n"
                f"Output Instruction:\n"
                f"1. 'thought_process': Describe the pattern, colors, and logic explicitly.\n"
                f"2. 'solution_grid': Provide the 2D output grid strictly as a list of lists e.g. [[0, 1], [2, 0]].\n"
            )

            try:
                output_state = worker(task_prompt, commit_mode='auto-reject')
                final_output_obj = output_state.get('final_output')

                parsed_grid = None
                reasoning = ""
                raw_dump = ""

                if final_output_obj and isinstance(final_output_obj, dict):
                    raw_grid = final_output_obj.get('solution_grid')
                    reasoning = final_output_obj.get('thought_process', '')
                    if isinstance(raw_grid, list):
                        parsed_grid = raw_grid
                    else:
                        parsed_grid = extract_grid_from_text(str(raw_grid))
                    raw_dump = final_output_obj
                else:
                    raw_dump = str(final_output_obj)
                    parsed_grid = extract_grid_from_text(raw_dump)
                    reasoning = "Parse failed, raw text only."

                is_correct, msg = check_answer(parsed_grid, test_output_gt)

                if not is_correct:
                    task_passed = False
                    # Only log warning if strictly needed to reduce noise
                    # logger.warning(f"❌ FAIL {task_id}: {msg}")

                task_logs.append(
                    {
                        'test_index': t_idx,
                        'passed': is_correct,
                        'msg': msg,
                        'pred': parsed_grid,
                        'gt': test_output_gt
                    }
                )

            except Exception as e:
                logger.error(f"Error in execution: {e}")
                task_passed = False
                task_logs.append({'error': str(e)})

        total_processed += 1
        if task_passed:
            total_passed += 1

        current_acc = (total_passed/total_processed)*100
        logger.info(f"📊 Acc: {current_acc:.2f}% | Task {task_id}: {'✅ PASS' if task_passed else '❌ FAIL'}")

        result_entry = {
            'task_id': task_id,
            'completion': task_logs[0].get('pred') if task_logs else None,
            'reasoning': reasoning,
            'passed': task_passed,
            'full_logs': task_logs,
            'raw_output': str(raw_dump)
        }
        results.append(result_entry)

        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

    final_acc = (total_passed/len(dataset))*100 if len(dataset) > 0 else 0
    logger.info(f'✅ ARC Benchmark completed. Final Accuracy: {final_acc:.2f}%')


if __name__ == '__main__':
    run_benchmark()