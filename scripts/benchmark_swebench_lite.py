import json
import os

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from gcri.graphs.gcri_unit import GCRI
from gcri.config import scope

DATASET_HF_ID = "princeton-nlp/SWE-bench_Lite"
BENCHMARK_DIR = 'benchmark_results/swebench_lite'


@scope
def get_preset_name(config):
    if config.get('custom_config_path'):
        return os.path.splitext(os.path.basename(config.custom_config_path))[0]
    return 'none'


RESULT_FILE = os.path.join(BENCHMARK_DIR, f'swebench_lite_results_{get_preset_name()}.json')


class SWEBenchResult(BaseModel):
    analysis: str = Field(
        ...,
        description='Analysis of the issue: what is the bug, what is the expected behavior, and what needs to be fixed.'
    )
    file_changes: str = Field(
        ...,
        description='Description of which files need to be modified and what changes are required.'
    )
    patch: str = Field(
        ...,
        description='The unified diff patch that fixes the issue. Format: standard git diff output.'
    )


def setup_directories():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


def normalize_patch(patch: str) -> str:
    if not patch:
        return ''
    lines = patch.strip().split('\n')
    normalized = []
    for line in lines:
        line = line.rstrip()
        if line.startswith('diff --git') or line.startswith('index '):
            continue
        if line.startswith('---') or line.startswith('+++'):
            parts = line.split('\t')[0]
            normalized.append(parts)
        elif line.startswith('@@') or line.startswith('+') or line.startswith('-') or line.startswith(' '):
            normalized.append(line)
    return '\n'.join(normalized)


def calculate_patch_similarity(pred_patch: str, gold_patch: str) -> tuple[float, str]:
    if not pred_patch:
        return 0.0, 'Empty patch'

    pred_lines = set(normalize_patch(pred_patch).split('\n'))
    gold_lines = set(normalize_patch(gold_patch).split('\n'))

    pred_additions = {l for l in pred_lines if l.startswith('+')}
    pred_deletions = {l for l in pred_lines if l.startswith('-')}
    gold_additions = {l for l in gold_lines if l.startswith('+')}
    gold_deletions = {l for l in gold_lines if l.startswith('-')}

    add_overlap = len(pred_additions & gold_additions)
    del_overlap = len(pred_deletions & gold_deletions)

    total_gold = len(gold_additions)+len(gold_deletions)
    total_pred = len(pred_additions)+len(pred_deletions)

    if total_gold == 0:
        return 0.0, 'No changes in gold patch'

    recall = (add_overlap+del_overlap)/total_gold if total_gold > 0 else 0
    precision = (add_overlap+del_overlap)/total_pred if total_pred > 0 else 0

    if precision+recall > 0:
        f1 = 2*(precision*recall)/(precision+recall)
    else:
        f1 = 0.0

    return f1, f'F1: {f1:.2%} (P: {precision:.2%}, R: {recall:.2%})'


@scope
def run_benchmark(config, num_samples=None):
    config.protocols.force_output = True
    logger.info(config.to_xyz())
    load_dotenv()
    setup_directories()

    logger.info('🤖 GCRI Worker Initializing for SWE-bench Lite...')
    worker = GCRI(config, schema=SWEBenchResult)

    logger.info(f'📚 Loading SWE-bench Lite dataset from {DATASET_HF_ID}...')
    try:
        dataset = load_dataset(DATASET_HF_ID, split='test')
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

    logger.info(f'📊 Loaded {len(dataset)} instances')

    if num_samples:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
        logger.info(f'🔍 Running on first {num_samples} samples.')

    results = []
    processed_ids = set()
    total_processed = 0
    total_f1_sum = 0.0

    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for item in existing_data:
                    t_id = item.get('instance_id')
                    if item.get('patch_generated'):
                        results.append(item)
                        processed_ids.add(t_id)
                        total_f1_sum += item.get('f1_score', 0)
                total_processed = len(results)
                logger.info(f'🔄 Resuming... {total_processed} items loaded.')
        except json.JSONDecodeError:
            logger.warning('⚠️ Result file is corrupt. Starting fresh.')

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc='SWE-bench Lite'):
        instance_id = item.get('instance_id', str(idx))
        if instance_id in processed_ids:
            continue

        try:
            problem_statement = item.get('problem_statement', '')
            repo = item.get('repo', '')
            gold_patch = item.get('patch', '')
            hints = item.get('hints_text', '')
            base_commit = item.get('base_commit', '')

            task_prompt = (
                f'You are an expert software engineer debugging a real-world GitHub issue.\n\n'
                f'Repository: {repo}\n'
                f'Base Commit: {base_commit}\n\n'
                f'--- ISSUE ---\n{problem_statement}\n\n'
            )

            if hints:
                task_prompt += f'--- HINTS FROM DISCUSSION ---\n{hints[:2000]}\n\n'

            task_prompt += (
                f'Your task:\n'
                f'1. Analyze the issue and identify the root cause\n'
                f'2. Determine which files need to be modified\n'
                f'3. Generate a unified diff patch that fixes the issue\n\n'
                f'The patch should be in standard git diff format:\n'
                f'```diff\n'
                f'--- a/path/to/file.py\n'
                f'+++ b/path/to/file.py\n'
                f'@@ -line,count +line,count @@\n'
                f' context line\n'
                f'-removed line\n'
                f'+added line\n'
                f'```'
            )

            logger.info(f'▶ Running: {instance_id} ({repo})')

            output_state = worker(task_prompt, commit_mode='auto-reject')
            final_output_obj = output_state.get('final_output')

            generated_patch = ''
            analysis = ''
            file_changes = ''

            if final_output_obj:
                if isinstance(final_output_obj, dict):
                    generated_patch = final_output_obj.get('patch', '')
                    analysis = final_output_obj.get('analysis', '')
                    file_changes = final_output_obj.get('file_changes', '')
                    raw_dump = final_output_obj
                else:
                    raw_dump = str(final_output_obj)
                    generated_patch = str(final_output_obj)
            else:
                raw_dump = 'No final output generated.'

            f1_score, eval_message = calculate_patch_similarity(generated_patch, gold_patch)

            total_processed += 1
            total_f1_sum += f1_score

            avg_f1 = total_f1_sum/total_processed

            logger.info(f'🧪 {instance_id} | {eval_message} | Avg F1: {avg_f1:.2%}')

            result = {
                'instance_id': instance_id,
                'repo': repo,
                'problem_statement': problem_statement[:500],
                'patch_generated': generated_patch,
                'patch_gold': gold_patch,
                'analysis': analysis,
                'file_changes': file_changes,
                'f1_score': f1_score,
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
            logger.error(f'❌ Error processing {instance_id}: {e}')
            continue

    final_avg_f1 = total_f1_sum/len(dataset) if len(dataset) > 0 else 0
    logger.info(f'✅ SWE-bench Lite completed. Average F1: {final_avg_f1:.2%}')
    logger.info(f'📄 Results saved to {RESULT_FILE}')
    logger.info(
        '⚠️ Note: This measures patch similarity. '
        'For official evaluation, run patches through SWE-bench harness with Docker.'
    )


if __name__ == '__main__':
    run_benchmark()
