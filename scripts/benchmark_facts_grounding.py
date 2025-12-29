import json
import os

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from gcri.graphs.gcri_unit import GCRI
from gcri.config import scope

DATASET_HF_ID = "google/FACTS-grounding-public"
BENCHMARK_DIR = 'benchmark_results/facts_grounding'


@scope
def get_preset_name(config):
    if config.get('custom_config_path'):
        return os.path.splitext(os.path.basename(config.custom_config_path))[0]
    return 'none'


RESULT_FILE = os.path.join(BENCHMARK_DIR, f'facts_results_{get_preset_name()}.json')


class FACTSResult(BaseModel):
    response: str = Field(
        ...,
        description='Your response to the user request, grounded exclusively in the provided document. Do not include any information not present in the document.'
    )


def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


@scope
def run_benchmark(config, num_samples=None):
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()

    logger.info('🤖 GCRI Worker Initializing for FACTS Grounding...')
    worker = GCRI(config, schema=FACTSResult)

    logger.info(f'📚 Loading FACTS Grounding dataset from {DATASET_HF_ID}...')
    try:
        dataset = load_dataset(DATASET_HF_ID, split='public')
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

    logger.info(f'📊 Loaded {len(dataset)} examples')

    if num_samples:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
        logger.info(f'🔍 Running on first {num_samples} samples.')

    results = []
    processed_ids = set()

    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                valid_results = []
                for item in existing_data:
                    t_id = item.get('task_id')
                    response = item.get('model_response')
                    if response:
                        valid_results.append(item)
                        processed_ids.add(str(t_id))
                results = valid_results
                logger.info(f'🔄 Resuming... {len(results)} valid items retained.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='FACTS Grounding'):
        task_id = str(idx)
        if task_id in processed_ids:
            continue

        try:
            document = item.get('context_document', '')
            user_request = item.get('user_request', '')
            system_instruction = item.get('system_instruction', '')

            if not document or not user_request:
                logger.warning(f'⚠️ Task {task_id}: Missing document or request')
                continue

            task_prompt = (
                f'{system_instruction}\n\n' if system_instruction else ''
            ) + (
                f'--- DOCUMENT START ---\n{document[:15000]}\n--- DOCUMENT END ---\n\n'
                f'User Request: {user_request}\n\n'
                f'Provide a thorough and accurate response grounded only in the document above.'
            )

            logger.info(f'▶ Running Task #{task_id}: {user_request[:60]}...')

            output_state = worker(task_prompt, commit_mode='auto-reject')

            model_response = ''
            raw_dump = 'No output generated.'

            if output_state is None:
                logger.warning(f'⚠️ Task {task_id}: GCRI returned None')
            else:
                final_output_obj = output_state.get('final_output')
                if final_output_obj:
                    if isinstance(final_output_obj, dict):
                        model_response = final_output_obj.get('response', '')
                        raw_dump = final_output_obj
                    else:
                        model_response = str(final_output_obj)
                        raw_dump = str(final_output_obj)

            logger.info(f'✅ Task #{task_id} completed. Response length: {len(model_response)}')

            result = {
                'task_id': task_id,
                'user_request': user_request[:500],
                'document_length': len(document),
                'model_response': model_response,
                'raw_output': raw_dump,
                'full_state': {
                    'best_branch': output_state.get('best_branch_index') if output_state else None,
                    'decision': output_state.get('decision') if output_state else None,
                    'iterations': output_state.get('count', 0) if output_state else 0
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

    logger.info(f'✅ FACTS Grounding completed. {len(results)} responses generated.')
    logger.info(f'📄 Results saved to {RESULT_FILE}')


if __name__ == '__main__':
    run_benchmark()
