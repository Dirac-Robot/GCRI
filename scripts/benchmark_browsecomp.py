import json
import os

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from gcri.graphs.gcri_unit import GCRI
from gcri.config import scope

DATASET_HF_ID = "openai/BrowseCompLongContext"
BENCHMARK_DIR = 'benchmark_results/browsecomp'


@scope
def get_preset_name(config):
    if config.get('custom_config_path'):
        return os.path.splitext(os.path.basename(config.custom_config_path))[0]
    return 'none'


RESULT_FILE = os.path.join(BENCHMARK_DIR, f'browsecomp_results_{get_preset_name()}.json')


class BrowseCompResult(BaseModel):
    reasoning: str = Field(
        ...,
        description='Your step-by-step reasoning process to find the answer, including search strategies and information synthesis.'
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
    answer = answer.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    return ' '.join(answer.split())


def check_answer(model_answer: str, ground_truth: str) -> tuple[bool, str]:
    if not ground_truth:
        return False, 'No ground truth available'

    model_norm = normalize_answer(model_answer)
    truth_norm = normalize_answer(ground_truth)

    if not model_norm:
        return False, 'Empty answer'

    if model_norm == truth_norm:
        return True, 'Exact Match'

    if truth_norm in model_norm or model_norm in truth_norm:
        return True, 'Substring Match'

    truth_words = set(truth_norm.split())
    model_words = set(model_norm.split())
    if truth_words and model_words:
        overlap = len(truth_words & model_words)/len(truth_words)
        if overlap >= 0.8:
            return True, f'High Overlap ({overlap:.0%})'

    return False, f'Mismatch'


@scope
def run_benchmark(config, num_samples=None):
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()

    logger.info('🤖 GCRI Worker Initializing for BrowseComp...')
    worker = GCRI(config, schema=BrowseCompResult)

    logger.info(f'📚 Loading BrowseComp dataset from {DATASET_HF_ID}...')
    try:
        dataset = load_dataset(DATASET_HF_ID, split='train')
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
    total_correct = 0

    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                valid_results = []
                for item in existing_data:
                    t_id = item.get('task_id')
                    answer = item.get('model_answer')
                    if answer:
                        valid_results.append(item)
                        processed_ids.add(str(t_id))
                results = valid_results
                total_processed = len(results)
                total_correct = sum(1 for item in results if item.get('is_correct', False))
                logger.info(f'🔄 Resuming... {total_processed} valid items retained.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='BrowseComp'):
        task_id = str(idx)
        if task_id in processed_ids:
            continue

        try:
            question = item.get('question', item.get('query', ''))
            ground_truth = item.get('answer', item.get('gold_answer', ''))

            task_prompt = (
                f'You are an expert web researcher with exceptional information-finding abilities.\n'
                f'Answer the following challenging question that requires deep web research.\n\n'
                f'Question: {question}\n\n'
                f'Think step by step about how you would search for and synthesize the information.\n'
                f'Provide a precise, factual answer.'
            )

            logger.info(f'▶ Running Task #{task_id}: {question[:60]}...')

            output_state = worker(task_prompt, commit_mode='auto-reject')
            final_output_obj = output_state.get('final_output')

            model_answer = ''
            model_reasoning = ''

            if final_output_obj:
                if isinstance(final_output_obj, dict):
                    model_answer = final_output_obj.get('final_answer', '')
                    model_reasoning = final_output_obj.get('reasoning', '')
                    raw_dump = final_output_obj
                else:
                    raw_dump = str(final_output_obj)
                    model_answer = str(final_output_obj)
            else:
                raw_dump = 'No final output generated.'

            is_correct, eval_message = check_answer(model_answer, ground_truth)

            total_processed += 1
            if is_correct:
                total_correct += 1

            current_accuracy = (total_correct/total_processed)*100

            status_icon = '✅ PASS' if is_correct else '❌ FAIL'
            logger.info(
                f'🧪 {status_icon} | {eval_message} | Acc: {current_accuracy:.2f}%'
            )

            result = {
                'task_id': task_id,
                'question': question[:500],
                'ground_truth': ground_truth,
                'model_answer': model_answer,
                'reasoning': model_reasoning[:500],
                'is_correct': is_correct,
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
            logger.error(f'❌ Error processing task {task_id}: {e}')
            continue

    final_acc = (total_correct/len(dataset))*100 if len(dataset) > 0 else 0
    logger.info(f'✅ BrowseComp completed. Final Accuracy: {final_acc:.2f}%')
    logger.info(f'📄 Results saved to {RESULT_FILE}')


if __name__ == '__main__':
    run_benchmark()
