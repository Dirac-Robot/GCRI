import json
import random
import os
from enum import Enum
from typing import Literal

from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm


MODEL_NAME = 'gemini-3-flash-preview'
THINKING_LEVEL = 'medium'
BENCHMARK_DIR = 'benchmark_results/gpqa'
RESULT_FILE = os.path.join(BENCHMARK_DIR, f'gpqa_diamond_results_{MODEL_NAME}_{THINKING_LEVEL}.json')
DATASET_NAME = 'idavidrein/gpqa'
DATASET_SUBSET = 'gpqa_diamond'


class GPQAResponse(BaseModel):
    selected_choice: Literal['A', 'B', 'C', 'D'] = Field(
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


def run_benchmark(num_samples=None):
    load_dotenv()
    setup_directories()
    logger.info(f'🤖 {MODEL_NAME} (thinking={THINKING_LEVEL}) Initializing for GPQA...')
    client = genai.Client()
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

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=task_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
                    response_mime_type='application/json',
                    response_schema=GPQAResponse
                )
            )

            raw_text = response.text or ''
            parsed_choice = ''
            try:
                parsed_obj = GPQAResponse.model_validate_json(raw_text)
                parsed_choice = parsed_obj.selected_choice
            except Exception as parse_err:
                logger.warning(f'⚠️ Failed to parse response: {parse_err}')

            thinking_content = ''
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        thinking_content += (part.text or '')

            is_correct = (parsed_choice == correct_letter)
            total_processed += 1
            if is_correct:
                total_correct += 1

            current_accuracy = (total_correct/total_processed)*100
            log_msg = '✅ CORRECT' if is_correct else f'❌ WRONG (Pred: {parsed_choice}, GT: {correct_letter})'
            logger.info(f'🧪 Result: {log_msg} | Acc: {current_accuracy:.2f}%')

            result_entry = {
                'question': question,
                'choices_map': choice_map,
                'correct_letter': correct_letter,
                'correct_answer_text': correct_answer,
                'model_choice': parsed_choice,
                'model_reasoning': thinking_content,
                'model_raw_output': raw_text,
                'is_correct': is_correct
            }
            results.append(result_entry)

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
