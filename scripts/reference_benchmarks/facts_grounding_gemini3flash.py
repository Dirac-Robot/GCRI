import json
import os

from datasets import load_dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from tqdm import tqdm

DATASET_HF_ID = "google/FACTS-grounding-public"
BENCHMARK_DIR = 'benchmark_results/facts_grounding'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'facts_results_gemini3.json')
MODEL_NAME = 'gemini-3-flash-preview'


def setup():
    load_dotenv()
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


def generate_response(llm, document: str, user_request: str, system_instruction: str = '') -> tuple[str, str]:
    task_prompt = (
        f'{system_instruction}\n\n' if system_instruction else ''
    ) + (
        f'--- DOCUMENT START ---\n{document[:15000]}\n--- DOCUMENT END ---\n\n'
        f'User Request: {user_request}\n\n'
        f'Provide a thorough and accurate response grounded only in the document above.'
    )

    try:
        response = llm.invoke(task_prompt)
        content = response.content if hasattr(response, 'content') else response
        if isinstance(content, list):
            response_text = ' '.join(str(c) for c in content)
        else:
            response_text = str(content)
        return response_text.strip(), response_text
    except Exception as e:
        logger.error(f'Generation error: {e}')
        return '', str(e)


def run_benchmark(num_samples=None):
    setup()

    logger.info(f'🤖 Initializing {MODEL_NAME}...')
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2)

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

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc=f'FACTS Grounding ({MODEL_NAME})'):
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

            logger.info(f'▶ Running Task #{task_id}: {user_request[:60]}...')

            model_response, raw_response = generate_response(llm, document, user_request, system_instruction)

            logger.info(f'✅ Task #{task_id} completed. Response length: {len(model_response)}')

            result = {
                'task_id': task_id,
                'user_request': user_request[:500],
                'document_length': len(document),
                'model_response': model_response,
                'raw_response': raw_response[:2000] if raw_response else '',
                'model': MODEL_NAME
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
