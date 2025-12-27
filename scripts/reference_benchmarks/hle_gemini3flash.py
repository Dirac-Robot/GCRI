import json
import os

from google import genai
from google.genai import types
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

BENCHMARK_DIR = 'benchmark_results/hle_text_only'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'hle_results_gemini3_medium.json')
MODEL_NAME = 'gemini-3-flash-preview'
THINKING_LEVEL = 'medium'


def setup():
    load_dotenv()
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


def normalize_answer(answer: str) -> str:
    if not answer:
        return ''
    answer = answer.strip().lower()
    answer = answer.replace('.', '').replace(',', '')
    return answer


def check_answer(parsed_answer: str, ground_truth: str) -> bool:
    norm_parsed = normalize_answer(parsed_answer)
    norm_truth = normalize_answer(ground_truth)
    if not norm_parsed or not norm_truth:
        return False
    if norm_parsed == norm_truth:
        return True
    if norm_truth in norm_parsed or norm_parsed in norm_truth:
        return True
    return False


def generate_answer(client, question: str) -> str:
    task_prompt = (
        f'You are taking "Humanity\'s Last Exam". Solve the following problem.\n'
        f'Question: {question}\n\n'
        f'Provide your answer directly and concisely. Just the final answer, no explanation needed.'
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=task_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
                temperature=0.7,
                max_output_tokens=2048,
            )
        )
        return response.text.strip() if response.text else ''
    except Exception as e:
        logger.error(f'Generation error: {e}')
        return ''


def run_benchmark(num_samples=None):
    setup()

    logger.info(f'🤖 Initializing {MODEL_NAME} with thinking_level={THINKING_LEVEL}...')
    client = genai.Client()

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
    total_processed = 0
    total_correct = 0

    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                results = existing_data
                processed_ids = {item['id'] for item in existing_data}
                total_processed = len(results)
                total_correct = sum(1 for item in results if item.get('is_correct', False))
                logger.info(f'🔄 Resuming... {len(results)} items already processed.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc=f'HLE ({MODEL_NAME})'):
        problem_id = str(item.get('id', idx))

        if problem_id in processed_ids:
            continue

        try:
            question = item.get('question', '')
            answer_key = item.get('answer', '')

            logger.info(f'▶ Running Task ID: {problem_id}')
            parsed_answer = generate_answer(client, question)

            is_correct = check_answer(parsed_answer, answer_key)

            total_processed += 1
            if is_correct:
                total_correct += 1

            current_accuracy = (total_correct/total_processed)*100
            logger.info(
                f'🧪 Result: {"✅ PASS" if is_correct else "❌ FAIL"} | Acc: {current_accuracy:.2f}%'
            )

            result_entry = {
                'id': problem_id,
                'question': question[:500],
                'ground_truth': answer_key,
                'parsed_answer': parsed_answer,
                'is_correct': is_correct,
                'model': MODEL_NAME,
                'thinking_level': THINKING_LEVEL
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

    final_acc = (total_correct/len(dataset))*100 if len(dataset) > 0 else 0
    logger.info(f'✅ HLE Benchmark completed. Final Accuracy: {final_acc:.2f}%')
    logger.info(f'📄 Results saved to {RESULT_FILE}')


if __name__ == '__main__':
    run_benchmark()
