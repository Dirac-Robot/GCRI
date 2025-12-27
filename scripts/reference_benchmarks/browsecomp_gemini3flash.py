import json
import os

from datasets import load_dataset
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from tqdm import tqdm


@tool
def search_web(query: str) -> str:
    """Search the web for information about a given query."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return 'No results found.'
            formatted = []
            for r in results:
                formatted.append(f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}")
            return '\n---\n'.join(formatted)
    except Exception as e:
        return f'Search Error: {e}'

DATASET_HF_ID = "openai/BrowseCompLongContext"
BENCHMARK_DIR = 'benchmark_results/browsecomp'
RESULT_FILE = os.path.join(BENCHMARK_DIR, 'browsecomp_results_gemini3.json')
MODEL_NAME = 'gemini-3-flash-preview'


def setup():
    load_dotenv()
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


def generate_answer(llm_with_tools, question: str, max_iterations: int = 5) -> tuple[str, str]:
    from langchain_core.messages import HumanMessage, ToolMessage

    task_prompt = (
        f'You are an expert web researcher with exceptional information-finding abilities.\n'
        f'Answer the following challenging question that requires deep web research.\n\n'
        f'Question: {question}\n\n'
        f'You have access to a search_web tool. Use it to search for information on the web.\n'
        f'Provide a precise, factual answer. Be concise and direct.'
    )

    tools_map = {'search_web': search_web}
    messages = [HumanMessage(content=task_prompt)]
    full_response = ''

    try:
        for _ in range(max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                content = response.content if hasattr(response, 'content') else response
                if isinstance(content, list):
                    full_response = ' '.join(str(c) for c in content)
                else:
                    full_response = str(content)
                return full_response.strip(), full_response

            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                call_id = tool_call['id']

                if tool_name in tools_map:
                    try:
                        tool_result = tools_map[tool_name].invoke(tool_args)
                        logger.debug(f'🔍 Tool {tool_name} called with {tool_args}')
                    except Exception as e:
                        tool_result = f'Tool Error: {e}'
                else:
                    tool_result = f'Unknown tool: {tool_name}'

                messages.append(ToolMessage(content=str(tool_result), tool_call_id=call_id, name=tool_name))

        return full_response.strip(), 'Max iterations reached'
    except Exception as e:
        logger.error(f'Generation error: {e}')
        return '', str(e)


def run_benchmark(num_samples=None):
    setup()

    logger.info(f'🤖 Initializing {MODEL_NAME} with search_web tool...')
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.7)
    llm_with_tools = llm.bind_tools([search_web])

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

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc=f'BrowseComp ({MODEL_NAME})'):
        task_id = str(idx)
        if task_id in processed_ids:
            continue

        try:
            question = item.get('question', item.get('query', ''))
            ground_truth = item.get('answer', item.get('gold_answer', ''))

            logger.info(f'▶ Running Task #{task_id}: {question[:60]}...')

            model_answer, raw_response = generate_answer(llm_with_tools, question)

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
                'model_answer': model_answer[:500],
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
            logger.error(f'❌ Error processing task {task_id}: {e}')
            continue

    final_acc = (total_correct/len(dataset))*100 if len(dataset) > 0 else 0
    logger.info(f'✅ BrowseComp completed. Final Accuracy: {final_acc:.2f}%')
    logger.info(f'📄 Results saved to {RESULT_FILE}')


if __name__ == '__main__':
    run_benchmark()
