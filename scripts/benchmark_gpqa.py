import json
import random
import os

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from gcri.graphs.gcri_unit import GCRI
from gcri.config import scope

BENCHMARK_DIR = 'benchmark_results/gpqa'

@scope
def get_preset_name(config):
    if config.get('custom_config_path'):
        return os.path.splitext(os.path.basename(config.custom_config_path))[0]
    return 'none'


RESULT_FILE = os.path.join(BENCHMARK_DIR, f'gpqa_diamond_results_{get_preset_name()}.json')

DATASET_NAME = 'idavidrein/gpqa'
DATASET_SUBSET = 'gpqa_diamond'


class GPQAResult(BaseModel):
    thought_process: str = Field(
        ...,
        description='Detailed reasoning steps to arrive at the correct answer. Analyze the scientific principles involved.'
    )
    selected_choice: str = Field(
        ...,
        description='The single letter (A, B, C, or D) corresponding to the correct answer.'
    )


def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


def format_choices(correct_answer, incorrect_answers):
    all_choices = [correct_answer]+incorrect_answers
    random.shuffle(all_choices)
    options = ['A', 'B', 'C', 'D']
    choice_map = {}
    correct_letter = None
    formatted_str = ''
    for idx, choice_text in enumerate(all_choices):
        letter = options[idx]
        formatted_str += f'{letter}) {choice_text}\n'
        choice_map[letter] = choice_text
        if choice_text == correct_answer:
            correct_letter = letter
    return formatted_str.strip(), correct_letter, choice_map


@scope
def run_benchmark(config, num_samples=None):
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()
    logger.info('🤖 GCRI Worker Initializing for GPQA (Research Reasoning Mode)...')
    worker = GCRI(config, schema=GPQAResult)
    logger.info(f'📚 Loading {DATASET_NAME} ({DATASET_SUBSET}) dataset...')
    try:
        dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, split='train')
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

    if num_samples:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
        logger.info(f'🔍 Running on first {num_samples} samples.')

    results = []
    processed_ids = set()
    total_processed = 0
    total_correct = 0
    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                valid_results = []
                for item in existing_data:
                    # Unlike HumanEval, we don't need task_id strictly, but we can use index or Question text as ID
                    q_text = item.get('question')
                    if q_text:
                        valid_results.append(item)
                        processed_ids.add(q_text)
                results = valid_results
                total_processed = len(results)
                total_correct = sum(1 for item in results if item.get('is_correct', False))
                logger.info(f'🔄 Resuming... {total_processed} items retained.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')
    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='Benchmarking GPQA'):
        question = item.get('Question')
        if question in processed_ids:
            continue
        correct_answer = item.get('Correct Answer')
        incorrect_answers = [
            item.get('Incorrect Answer 1'),
            item.get('Incorrect Answer 2'),
            item.get('Incorrect Answer 3')
        ]
        if not correct_answer or not question:
            continue
        choices_str, correct_letter, choice_map = format_choices(correct_answer, incorrect_answers)

        try:
            task_prompt = (
                f'You are a PhD-level scientific researcher.\n'
                f'Answer the following question by reasoning through the scientific principles step-by-step.\n'
                f'The question is designed to be difficult and requires deep domain knowledge.\n\n'
                f'Question: {question}\n\n'
                f'Choices:\n{choices_str}\n\n'
                f'Select the single best answer (A, B, C, or D).'
            )
            logger.info(f'▶ Running Question: {question[:50]}...')
            output_state = worker(task_prompt, commit_mode='auto-reject')
            final_output_obj = output_state.get('final_output')

            parsed_choice = ''
            parsed_reasoning = ''

            if final_output_obj:
                if isinstance(final_output_obj, dict):
                    parsed_choice = final_output_obj.get('selected_choice', '').strip().upper()
                    parsed_choice = parsed_choice.replace(')', '').replace('.', '')
                    parsed_reasoning = final_output_obj.get('thought_process', '')
                    raw_dump = final_output_obj
                else:
                    raw_dump = str(final_output_obj)
            else:
                raw_dump = 'No final output generated.'
            is_correct = (parsed_choice == correct_letter)

            total_processed += 1
            if is_correct:
                total_correct += 1

            current_accuracy = (total_correct/total_processed)*100

            log_msg = '✅ CORRECT' if is_correct else f'❌ WRONG (Pred: {parsed_choice}, GT: {correct_letter})'
            logger.info(f'🧪 Result: {log_msg} | Acc: {current_accuracy:.2f}%')

            result_entry = {
                'question': question,
                'choices_map': choice_map,  # Save what A/B/C/D meant for this run
                'correct_letter': correct_letter,
                'correct_answer_text': correct_answer,
                'model_choice': parsed_choice,
                'model_reasoning': parsed_reasoning,
                'is_correct': is_correct,
                'raw_output': raw_dump,
                'full_state': {
                    'best_branch': output_state.get('best_branch_index'),
                    'iterations': output_state.get('count', 0)
                }
            }

            results.append(result_entry)

            # Save incrementally
            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except KeyboardInterrupt:
            logger.warning('⛔ Benchmark interrupted by user.')
            break
        except Exception as e:
            logger.error(f'❌ Error processing sample: {e}')
            continue

    final_acc = (total_correct/total_processed*100) if total_processed > 0 else 0
    logger.info(f'✅ Benchmark completed. Final Accuracy: {final_acc:.2f}%')
    logger.info(f'📄 Detailed results saved to {RESULT_FILE}')


if __name__ == '__main__':
    run_benchmark()