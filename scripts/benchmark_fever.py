import json
import os

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from gcri.graphs.gcri_unit import GCRI
from gcri.config import scope

DATASET_HF_ID = "fever/fever"
BENCHMARK_DIR = 'benchmark_results/fever'
VALID_LABELS = {'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'}


@scope
def get_preset_name(config):
    if config.get('custom_config_path'):
        return os.path.splitext(os.path.basename(config.custom_config_path))[0]
    return 'none'


RESULT_FILE = os.path.join(BENCHMARK_DIR, f'fever_results_{get_preset_name()}.json')


class FEVERResult(BaseModel):
    reasoning: str = Field(
        ...,
        description='Your step-by-step reasoning about the claim. Analyze what evidence would be needed to verify or refute the claim.'
    )
    verdict: str = Field(
        ...,
        description='Your verdict: must be exactly one of "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO".'
    )


def setup_directories():
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


@scope
def run_benchmark(config, num_samples=None, split='labelled_dev'):
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()

    logger.info('🤖 GCRI Worker Initializing for FEVER (Fact Verification)...')
    worker = GCRI(config, schema=FEVERResult)

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

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='FEVER Benchmark'):
        claim_id = str(item.get('id', idx))
        if claim_id in processed_ids:
            continue

        try:
            claim = item.get('claim', '')
            ground_truth = item.get('label', '')

            task_prompt = (
                f'You are a fact-checking expert. Analyze the following claim and determine its veracity.\n\n'
                f'Claim: "{claim}"\n\n'
                f'Based on your knowledge, classify this claim as:\n'
                f'- SUPPORTS: The claim is factually correct and can be verified\n'
                f'- REFUTES: The claim is factually incorrect\n'
                f'- NOT ENOUGH INFO: There is insufficient information to verify or refute the claim\n\n'
                f'Provide your reasoning and verdict.'
            )

            logger.info(f'▶ Running Claim #{claim_id}: {claim[:60]}...')

            output_state = worker(task_prompt, commit_mode='auto-reject')
            final_output_obj = output_state.get('final_output')

            model_verdict = ''
            model_reasoning = ''

            if final_output_obj:
                if isinstance(final_output_obj, dict):
                    model_verdict = final_output_obj.get('verdict', '')
                    model_reasoning = final_output_obj.get('reasoning', '')
                    raw_dump = final_output_obj
                else:
                    raw_dump = str(final_output_obj)
                    model_verdict = str(final_output_obj)
            else:
                raw_dump = 'No final output generated.'

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
                'reasoning': model_reasoning,
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
            logger.error(f'❌ Error processing claim {claim_id}: {e}')
            continue

    final_acc = (total_correct/len(dataset))*100 if len(dataset) > 0 else 0
    logger.info(f'✅ FEVER Benchmark completed. Final Accuracy: {final_acc:.2f}%')
    logger.info(f'📄 Detailed results saved to {RESULT_FILE}')


if __name__ == '__main__':
    run_benchmark()
