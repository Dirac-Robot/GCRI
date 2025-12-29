"""
FACTS Grounding Official Evaluation Script
Based on official evaluation methodology from Google DeepMind.

Two-stage evaluation:
1. Eligibility Filter: Check if response follows instructions
2. Factual Accuracy: Sentence-level grounding check (supported/unsupported/contradictory/no_rad)
"""
import json
import os
from enum import Enum

from datasets import load_dataset
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

DATASET_HF_ID = "google/FACTS-grounding-public"
BENCHMARK_DIR = 'benchmark_results/facts_grounding'
JUDGE_MODEL = 'gemini-3-pro'


class InstructionFollowingVerdict(str, Enum):
    NO_ISSUES = "No Issues"
    MINOR_ISSUES = "Minor Issue(s)"
    MAJOR_ISSUES = "Major Issue(s)"


class EligibilityResult(BaseModel):
    analysis: str = Field(..., description='Detailed analysis of instruction following.')
    verdict: InstructionFollowingVerdict = Field(..., description='Final verdict for instruction following.')


class SentenceLabel(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTORY = "contradictory"
    NO_RAD = "no_rad"


class SentenceEvaluation(BaseModel):
    sentence: str = Field(..., description='The sentence being analyzed.')
    label: SentenceLabel = Field(..., description='The grounding label.')
    rationale: str = Field(..., description='Brief explanation for the assigned label.')
    excerpt: str | None = Field(None, description='Relevant excerpt from context (for supported/contradictory).')


class FactualAccuracyResult(BaseModel):
    sentence_evaluations: list[SentenceEvaluation] = Field(..., description='Evaluation for each sentence.')
    final_verdict: str = Field(..., description='Accurate or Inaccurate')


def setup():
    load_dotenv()


def load_results(result_file: str) -> list:
    if not os.path.exists(result_file):
        logger.error(f'Result file not found: {result_file}')
        return []
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_dataset_examples() -> dict:
    dataset = load_dataset(DATASET_HF_ID, split='public')
    examples = {}
    for idx, item in enumerate(dataset):
        examples[str(idx)] = {
            'context_document': item.get('context_document', ''),
            'user_request': item.get('user_request', ''),
            'system_instruction': item.get('system_instruction', '')
        }
    return examples


def check_eligibility(llm, user_request: str, response: str) -> EligibilityResult:
    prompt = f'''Your task is to analyze the response based on the criterion of "Instruction Following".

**Instruction Following**
Please first list the instructions in the user query.
In general, an instruction is VERY important if it is specifically asked for in the prompt and deviates from the norm.
After listing the instructions, rank them in order of importance.
Then check if the response meets each of the instructions.
For each instruction, determine whether the response meets, partially meets, or does not meet the requirement.

## User query
<|begin_of_query|>
{user_request}
<|end_of_query|>

## Response:
<|begin_of_response|>
{response}
<|end_of_response|>

Provide your analysis and final verdict.'''

    try:
        return llm.invoke(prompt)
    except Exception as e:
        logger.error(f'Eligibility check error: {e}')
        return None


def check_factual_accuracy(llm, user_request: str, context_document: str, response: str) -> FactualAccuracyResult:
    prompt = f'''You are an expert fact-checker analyzing if sentences are grounded in context.

**Guidelines:**
1. For each sentence in the response, determine if it is:
   - `supported`: Directly supported by evidence in the context
   - `unsupported`: Cannot be verified from the context (hallucination)
   - `contradictory`: Directly contradicts the context
   - `no_rad`: General expressions that don't require factual attribution

2. Be STRICT. Unless you find clear evidence in the context, mark as `unsupported`.
3. Do not use world knowledge unless truly trivial.

**User Query:**
{user_request}

**Context:**
{context_document[:12000]}

**Response:**
{response}

Analyze each sentence and provide the final verdict.
If ANY sentence is contradictory or unsupported (except no_rad), the final verdict is "Inaccurate".
Otherwise, "Accurate".'''

    try:
        return llm.invoke(prompt)
    except Exception as e:
        logger.error(f'Factual accuracy check error: {e}')
        return None


def calculate_scores(eval_result: FactualAccuracyResult) -> dict:
    if not eval_result or not eval_result.sentence_evaluations:
        return {'supported': 0, 'unsupported': 0, 'contradictory': 0, 'no_rad': 0, 'accuracy': 0.0}

    counts = {'supported': 0, 'unsupported': 0, 'contradictory': 0, 'no_rad': 0}
    for sent in eval_result.sentence_evaluations:
        counts[sent.label.value] += 1

    total_rad = counts['supported']+counts['unsupported']+counts['contradictory']
    accuracy = counts['supported']/total_rad if total_rad > 0 else 1.0

    return {**counts, 'accuracy': accuracy}


def run_evaluation(result_file: str, output_file: str = None):
    setup()

    if output_file is None:
        base_name = os.path.splitext(result_file)[0]
        output_file = f'{base_name}_judged_official.json'

    logger.info(f'🧑‍⚖️ Initializing {JUDGE_MODEL} as judge...')
    eligibility_llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0).with_structured_output(EligibilityResult)
    factual_llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0).with_structured_output(FactualAccuracyResult)

    logger.info('📚 Loading dataset examples...')
    dataset_examples = load_dataset_examples()

    logger.info(f'📄 Loading results from {result_file}...')
    results = load_results(result_file)
    if not results:
        return

    logger.info(f'📊 Evaluating {len(results)} responses...')

    judged_results = []
    processed_ids = set()

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                judged_results = existing
                processed_ids = {item['task_id'] for item in existing}
                logger.info(f'🔄 Resuming... {len(judged_results)} already judged.')
        except json.JSONDecodeError:
            pass

    stats = {'eligible': 0, 'ineligible': 0, 'accurate': 0, 'inaccurate': 0, 'total_accuracy': 0.0}
    for r in judged_results:
        if r.get('is_eligible'):
            stats['eligible'] += 1
            if r.get('is_accurate'):
                stats['accurate'] += 1
            else:
                stats['inaccurate'] += 1
            stats['total_accuracy'] += r.get('sentence_accuracy', 0)
        else:
            stats['ineligible'] += 1

    for item in tqdm(results, desc='Judging (Official)'):
        task_id = item.get('task_id')
        if task_id in processed_ids:
            continue

        model_response = item.get('model_response', '')
        if not model_response:
            continue

        example = dataset_examples.get(task_id, {})
        document = example.get('context_document', '')
        user_request = example.get('user_request', item.get('user_request', ''))

        if not document:
            logger.warning(f'⚠️ Task {task_id}: No document found')
            continue

        logger.info(f'▶ Judging Task #{task_id}...')

        eligibility = check_eligibility(eligibility_llm, user_request, model_response)

        is_eligible = eligibility and eligibility.verdict != InstructionFollowingVerdict.MAJOR_ISSUES

        judged_item = {
            'task_id': task_id,
            'user_request': user_request[:200],
            'model_response': model_response[:500],
            'is_eligible': is_eligible,
            'eligibility_verdict': eligibility.verdict.value if eligibility else 'Error',
            'eligibility_analysis': eligibility.analysis[:300] if eligibility else '',
        }

        if is_eligible:
            stats['eligible'] += 1
            factual = check_factual_accuracy(factual_llm, user_request, document, model_response)

            if factual:
                scores = calculate_scores(factual)
                is_accurate = factual.final_verdict.lower() == 'accurate'

                logger.info(
                    f'   📈 Sentences: {len(factual.sentence_evaluations)} | '
                    f'Supported: {scores["supported"]} | Unsupported: {scores["unsupported"]} | '
                    f'Contradictory: {scores["contradictory"]} | Score: {scores["accuracy"]:.2f}'
                )

                judged_item.update({
                    'is_accurate': is_accurate,
                    'final_verdict': factual.final_verdict,
                    'sentence_accuracy': scores['accuracy'],
                    'supported_count': scores['supported'],
                    'unsupported_count': scores['unsupported'],
                    'contradictory_count': scores['contradictory'],
                    'no_rad_count': scores['no_rad'],
                    'sentence_evaluations': [
                        {'sentence': s.sentence[:100], 'label': s.label.value, 'rationale': s.rationale[:100]}
                        for s in factual.sentence_evaluations[:5]
                    ]
                })

                if is_accurate:
                    stats['accurate'] += 1
                else:
                    stats['inaccurate'] += 1
                stats['total_accuracy'] += scores['accuracy']
            else:
                judged_item['is_accurate'] = False
                judged_item['final_verdict'] = 'Error'
                stats['inaccurate'] += 1
        else:
            stats['ineligible'] += 1
            judged_item['is_accurate'] = False
            judged_item['final_verdict'] = 'Ineligible'

        judged_results.append(judged_item)

        n_eligible = stats['eligible']
        n_total = len(judged_results)
        if n_eligible > 0:
            facts_score = stats['accurate']/n_eligible*100
            logger.info(
                f'📊 FACTS Score: {facts_score:.1f}% ({stats["accurate"]}/{n_eligible} accurate) | '
                f'Eligible: {n_eligible}/{n_total}'
            )
        else:
            logger.info(f'📊 Eligible: 0/{n_total} | Ineligible: {stats["ineligible"]}')

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(judged_results, f, indent=4, ensure_ascii=False)

    n = len(judged_results)
    n_eligible = stats['eligible']
    if n > 0:
        logger.info('\n✅ Official FACTS Evaluation Complete!')
        logger.info('📊 Final Results:')
        logger.info(f'   - Eligible Rate: {n_eligible/n*100:.2f}% ({n_eligible}/{n})')
        if n_eligible > 0:
            logger.info(f'   - Accuracy Rate (eligible only): {stats["accurate"]/n_eligible*100:.2f}%')
            logger.info(f'   - Avg Sentence Accuracy: {stats["total_accuracy"]/n_eligible*100:.2f}%')
        logger.info(f'📄 Results saved to {output_file}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        logger.error('Usage: python evaluate_facts_grounding.py <result_file.json>')
        sys.exit(1)
    run_evaluation(sys.argv[1])
