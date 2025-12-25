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

DATASET_HF_ID = "basicv8vc/SimpleQA"
BENCHMARK_DIR = 'benchmark_results/simpleqa'


@scope
def get_preset_name(config):
    if config.get('custom_config_path'):
        return os.path.splitext(os.path.basename(config.custom_config_path))[0]
    return 'none'


RESULT_FILE = os.path.join(BENCHMARK_DIR, f'simpleqa_results_{get_preset_name()}.json')


class SimpleQAResult(BaseModel):
    thought_process: str = Field(
        ...,
        description='Your reasoning process to find the factual answer. Include any relevant knowledge you are drawing from.'
    )
    final_answer: str = Field(
        ...,
        description='The concise, factual answer to the question. Be precise and direct.'
    )


def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


def normalize_answer(answer: str) -> str:
    if not answer:
        return ''
    answer = answer.strip().lower()
    answer = re.sub(r'^(the|a|an)\s+', '', answer)
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = ' '.join(answer.split())
    return answer


def evaluate_answer(ground_truth: str, model_answer: str) -> tuple[bool, str]:
    if not ground_truth:
        return False, 'Error: No ground truth available'

    gt_normalized = normalize_answer(ground_truth)
    model_normalized = normalize_answer(model_answer)

    if not model_normalized:
        return False, 'Empty answer'

    if gt_normalized == model_normalized:
        return True, 'Exact Match'

    if gt_normalized in model_normalized or model_normalized in gt_normalized:
        return True, 'Substring Match'

    gt_words = set(gt_normalized.split())
    model_words = set(model_normalized.split())

    if gt_words and model_words:
        overlap = len(gt_words & model_words)/len(gt_words)
        if overlap >= 0.8:
            return True, f'High Overlap ({overlap:.0%})'

    return False, f'Mismatch (Expected: {ground_truth})'


@scope
def run_benchmark(config, num_samples=None):
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()

    logger.info('🤖 GCRI Worker Initializing for SimpleQA (Factuality Benchmark)...')
    worker = GCRI(config, schema=SimpleQAResult)

    logger.info(f'📚 Loading SimpleQA dataset from {DATASET_HF_ID}...')
    try:
        dataset = load_dataset(DATASET_HF_ID, split='test')
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

    logger.info(f'📊 Loaded {len(dataset)} questions')

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
                    if comp:
                        valid_results.append(item)
                        processed_ids.add(str(t_id))
                results = valid_results
                total_processed = len(results)
                total_passed = sum(1 for item in results if item.get('passed', False))
                logger.info(f'🔄 Resuming... {total_processed} valid items retained.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='SimpleQA Benchmark'):
        task_id = str(idx)
        if task_id in processed_ids:
            continue

        try:
            question = item.get('problem', item.get('question', ''))
            ground_truth = item.get('answer', '')
            metadata = item.get('metadata', {})
            topic = metadata.get('topic', 'General') if isinstance(metadata, dict) else 'General'

            task_prompt = (
                f'You are an expert knowledge assistant with access to factual information.\n'
                f'Answer the following question accurately and concisely.\n'
                f'Only provide the specific fact requested - do not add unnecessary context.\n\n'
                f'Question: {question}\n\n'
                f'Provide your reasoning and the precise factual answer.'
            )

            logger.info(f'▶ Running Task #{task_id}: {question[:60]}...')

            output_state = worker(task_prompt, commit_mode='auto-reject')
            final_output_obj = output_state.get('final_output')

            parsed_answer = ''
            parsed_reasoning = ''

            if final_output_obj:
                if isinstance(final_output_obj, dict):
                    parsed_answer = final_output_obj.get('final_answer', '')
                    parsed_reasoning = final_output_obj.get('thought_process', '')
                    raw_dump = final_output_obj
                else:
                    raw_dump = str(final_output_obj)
                    parsed_answer = str(final_output_obj)
            else:
                raw_dump = 'No final output generated.'

            is_passed, eval_message = evaluate_answer(ground_truth, parsed_answer)

            total_processed += 1
            if is_passed:
                total_passed += 1

            current_accuracy = (total_passed/total_processed)*100

            status_icon = '✅ PASS' if is_passed else '❌ FAIL'
            logger.info(
                f'🧪 {status_icon} | {eval_message} | Acc: {current_accuracy:.2f}%'
            )

            result = {
                'task_id': task_id,
                'question': question,
                'topic': topic,
                'ground_truth': ground_truth,
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

    final_acc = (total_passed/len(dataset))*100 if len(dataset) > 0 else 0
    logger.info(f'✅ SimpleQA Benchmark completed. Final Accuracy: {final_acc:.2f}%')
    logger.info(f'📄 Detailed results saved to {RESULT_FILE}')


if __name__ == '__main__':
    run_benchmark()
