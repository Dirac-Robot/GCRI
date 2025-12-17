import os
import json
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

from gcri.config import scope
from gcri.graphs.gcri_unit import GCRI

BENCHMARK_DIR = 'benchmark_results/hle_text_only'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'results.json')


class HLEResult(BaseModel):
    explanation: str = Field(
        ...,
        description='A detailed step-by-step explanation and reasoning for the answer choice.'
    )
    answer: str = Field(
        ...,
        description='The specific chosen answer (e.g., option letter, number, or short phrase).'
    )
    confidence: str = Field(
        ...,
        description='Confidence score between 0% and 100% (e.g., "95%").'
    )


def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


@scope
def run_benchmark(config, num_samples=None):
    load_dotenv()
    setup_directories()

    logger.info('🤖 GCRI Worker Initializing for HLE Benchmark (Text Only) with Custom Schema...')
    worker = GCRI(config, schema=HLEResult)

    logger.info('📚 Loading Humanity\'s Last Exam dataset...')
    try:
        dataset = load_dataset('cais/hle', split='test')
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

    logger.info(f'📊 Original dataset size: {len(dataset)}')

    def is_text_only(example):
        img = example.get('image')
        return img is None or (isinstance(img, str) and img.strip() == '')

    dataset = dataset.filter(is_text_only)
    logger.info(f'📉 Filtered text-only dataset size: {len(dataset)}')

    if num_samples:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
        logger.info(f'🔍 Running on first {num_samples} samples.')

    results = []
    processed_ids = set()

    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                results = existing_data
                processed_ids = {item['id'] for item in existing_data}
                logger.info(f'🔄 Resuming... {len(results)} items already processed.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='Benchmarking'):
        problem_id = str(item.get('id', idx))

        if problem_id in processed_ids:
            continue

        try:
            question = item.get('question', '')
            answer_key = item.get('answer', '')

            task_prompt = (
                f'You are taking "Humanity\'s Last Exam". Solve the following problem.\n'
                f'Question: {question}\n\n'
                f'Provide a robust explanation and the final answer.'
            )

            logger.info(f'▶ Running Task ID: {problem_id}')

            output_state = worker(task_prompt, auto_commit=True)
            final_output_obj = output_state.get('final_output')

            parsed_answer = ''
            parsed_explanation = ''
            parsed_confidence = ''
            raw_dump = None

            if final_output_obj:
                if isinstance(final_output_obj, dict):
                    parsed_answer = final_output_obj.get('answer', '')
                    parsed_explanation = final_output_obj.get('explanation', '')
                    parsed_confidence = final_output_obj.get('confidence', '')
                    raw_dump = final_output_obj
                else:
                    raw_dump = str(final_output_obj)
            else:
                raw_dump = 'No final output generated.'

            result_entry = {
                'id': problem_id,
                'question': question,
                'ground_truth': answer_key,
                'raw_output': raw_dump,
                'parsed_answer': parsed_answer,
                'parsed_explanation': parsed_explanation,
                'confidence': parsed_confidence,
                'full_state': {
                    'best_branch': output_state.get('best_branch_index'),
                    'decision': output_state.get('decision'),
                    'iterations': output_state.get('count', 0)
                }
            }
            results.append(result_entry)

            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except KeyboardInterrupt:
            logger.warning('⛔ Benchmark interrupted by user.')
            break
        except Exception as e:
            logger.error(f'❌ Error processing sample {problem_id}: {e}')
            continue

    logger.info(f'✅ Benchmark completed. Results saved to {RESULT_FILE}')


if __name__ == '__main__':
    run_benchmark()
