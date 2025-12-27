import json
import os

from datasets import load_dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from tqdm import tqdm

DATASET_HF_ID = "fever/fever"
BENCHMARK_DIR = 'benchmark_results/fever'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'fever_results_gemini3_with_search.json')
MODEL_NAME = 'gemini-3-flash-preview'
VALID_LABELS = {'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'}


def setup():
    load_dotenv()
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


def normalize_verdict(verdict: str) -> str:
    if not verdict:
        return ''
    verdict = verdict.strip().upper()
    if 'SUPPORT' in verdict:
        return 'SUPPORTS'
    if 'REFUTE' in verdict:
        return 'REFUTES'
    if 'NOT ENOUGH' in verdict or 'NEI' in verdict or 'INSUFFICIENT' in verdict:
        return 'NOT ENOUGH INFO'
    return verdict


def evaluate_verdict(ground_truth: str, model_verdict: str) -> tuple[bool, str]:
    if not ground_truth:
        return False, 'No ground truth available'

    gt_normalized = normalize_verdict(ground_truth)
    model_normalized = normalize_verdict(model_verdict)

    if not model_normalized:
        return False, 'Empty verdict'

    if model_normalized not in VALID_LABELS:
        return False, f'Invalid verdict: {model_normalized}'

    if gt_normalized == model_normalized:
        return True, f'Correct ({model_normalized})'

    return False, f'Wrong (Expected: {gt_normalized}, Got: {model_normalized})'


def extract_verdict_from_response(response_text: str) -> str:
    response_upper = response_text.upper()
    if 'SUPPORTS' in response_upper:
        return 'SUPPORTS'
    if 'REFUTES' in response_upper:
        return 'REFUTES'
    if 'NOT ENOUGH INFO' in response_upper or 'NOT ENOUGH INFORMATION' in response_upper:
        return 'NOT ENOUGH INFO'
    return response_text.strip()


def generate_verdict(llm_with_tools, claim: str) -> tuple[str, str]:
    task_prompt = (
        f'You are a fact-checking expert. Analyze the following claim and determine its veracity.\n\n'
        f'Claim: "{claim}"\n\n'
        f'After your analysis, classify this claim as exactly one of:\n'
        f'- SUPPORTS: The claim is factually correct\n'
        f'- REFUTES: The claim is factually incorrect\n'
        f'- NOT ENOUGH INFO: Insufficient information to verify\n\n'
        f'End your response with your final verdict on a new line, like: "Verdict: SUPPORTS"'
    )

    try:
        response = llm_with_tools.invoke(task_prompt)
        content = response.content if hasattr(response, 'content') else response
        if isinstance(content, list):
            response_text = ' '.join(str(c) for c in content)
        else:
            response_text = str(content)
        verdict = extract_verdict_from_response(response_text)
        return verdict, response_text
    except Exception as e:
        logger.error(f'Generation error: {e}')
        return '', str(e)


def run_benchmark(num_samples=None, split='labelled_dev'):
    setup()

    logger.info(f'🤖 Initializing {MODEL_NAME}...')

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.7)
    tools = []
    llm_with_tools = llm.bind_tools(tools)

    logger.info(f'📚 Loading FEVER dataset from {DATASET_HF_ID} (split: {split})...')
    try:
        dataset = load_dataset(DATASET_HF_ID, 'v1.0', split=split)
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

    logger.info(f'📊 Loaded {len(dataset)} claims')

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
                    t_id = item.get('claim_id')
                    verdict = item.get('model_verdict')
                    if verdict:
                        valid_results.append(item)
                        processed_ids.add(str(t_id))
                results = valid_results
                total_processed = len(results)
                total_correct = sum(1 for item in results if item.get('is_correct', False))
                logger.info(f'🔄 Resuming... {total_processed} valid items retained.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc=f'FEVER ({MODEL_NAME})'):
        claim_id = str(item.get('id', idx))
        if claim_id in processed_ids:
            continue

        try:
            claim = item.get('claim', '')
            ground_truth = item.get('label', '')

            logger.info(f'▶ Running Claim #{claim_id}: {claim[:60]}...')

            model_verdict, raw_response = generate_verdict(llm_with_tools, claim)

            is_correct, eval_message = evaluate_verdict(ground_truth, model_verdict)

            total_processed += 1
            if is_correct:
                total_correct += 1

            current_accuracy = (total_correct/total_processed)*100

            status_icon = '✅ PASS' if is_correct else '❌ FAIL'
            logger.info(
                f'🧪 {status_icon} | {eval_message} | Acc: {current_accuracy:.2f}%'
            )

            result = {
                'claim_id': claim_id,
                'claim': claim,
                'ground_truth': ground_truth,
                'model_verdict': normalize_verdict(model_verdict),
                'raw_response': raw_response[:1000] if raw_response else '',
                'is_correct': is_correct,
                'eval_message': eval_message,
                'model': MODEL_NAME
            }
            results.append(result)

            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except KeyboardInterrupt:
            logger.warning('⛔ Benchmark interrupted by user.')
            break
        except Exception as e:
            logger.error(f'❌ Error processing claim {claim_id}: {e}')
            continue

    final_acc = (total_correct/len(dataset))*100 if len(dataset) > 0 else 0
    logger.info(f'✅ FEVER Benchmark completed. Final Accuracy: {final_acc:.2f}%')
    logger.info(f'📄 Detailed results saved to {RESULT_FILE}')


if __name__ == '__main__':
    run_benchmark()
