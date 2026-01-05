from typing import Optional, Callable
import re
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class CodeOutput(BaseModel):
    code: str = Field(description='The complete, executable Python code that solves the given task.')
    confidence: float = Field(default=1.0, description='Confidence score for the solution.')


class TextOutput(BaseModel):
    answer: str = Field(description='The final answer to the question.')
    confidence: float = Field(default=1.0, description='Confidence score for the answer.')


def unescape_code(text: str) -> str:
    text = text.replace('\\n', '\n')
    text = text.replace('\\t', '\t')
    text = text.replace('\\r', '\r')
    text = text.replace("\\'", "'")
    text = text.replace('\\"', '"')
    return text


def extract_code_from_markdown(text: str) -> str:
    code_block_pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        return '\n'.join(matches)
    return text


BIGCODEBENCH_STANDARD_IMPORTS = '''import os
import re
import sys
import math
import json
import random
import string
import collections
import itertools
import functools
import datetime
import numpy as np
import pandas as pd
'''


def postprocess_bigcodebench(output: str) -> str:
    output = unescape_code(output)
    output = extract_code_from_markdown(output)
    output = output.strip()
    lines = output.split('\n')
    import_lines = []
    function_lines = []
    in_function = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')) and not in_function:
            import_lines.append(line)
        elif stripped.startswith('def task_func'):
            in_function = True
            function_lines.append(line)
        elif in_function:
            function_lines.append(line)
    if function_lines:
        result_parts = [BIGCODEBENCH_STANDARD_IMPORTS.strip()]
        if import_lines:
            result_parts.append('\n'.join(import_lines))
        result_parts.append('\n'.join(function_lines))
        return '\n\n'.join(result_parts)
    return BIGCODEBENCH_STANDARD_IMPORTS+output


def postprocess_humaneval(output: str) -> str:
    output = unescape_code(output)
    return extract_code_from_markdown(output).strip()


def postprocess_qa(output: str) -> str:
    return output.strip()


@dataclass
class BenchmarkConfig:
    schema: Optional[type[BaseModel]]
    prompt_prefix: str = ''
    prompt_suffix: str = ''
    post_process: Callable[[str], str] = field(default=lambda x: x)


CODE_BENCHMARK_PREFIX = '''You are solving a coding benchmark task. Your goal is to write correct, executable Python code.

CRITICAL INSTRUCTIONS:
1. Your final output MUST be the actual Python code, not a description of what the code does.
2. The code should be complete and directly executable.
3. Do not include markdown code blocks or explanations in the final output - only pure Python code.

'''

CODE_BENCHMARK_SUFFIX = '''

Remember: Return ONLY the Python code in your final output. No explanations, no markdown formatting.'''

BIGCODEBENCH_PREFIX = '''You are solving a BigCodeBench coding task. You will be given a function signature with a docstring.
Your task is to COMPLETE the function implementation.

CRITICAL INSTRUCTIONS:
1. The function MUST be named exactly as specified (usually 'task_func').
2. Include all necessary imports at the top.
3. Return the COMPLETE function definition including the 'def' line.
4. Do NOT include markdown code blocks - output pure Python code only.
5. Ensure the implementation follows the docstring specification exactly.

'''

QA_BENCHMARK_PREFIX = '''You are solving a question-answering benchmark task. Provide a concise and accurate answer.

'''

BENCHMARK_REGISTRY: dict[str, BenchmarkConfig] = {
    'BigCodeBench': BenchmarkConfig(
        schema=CodeOutput,
        prompt_prefix=BIGCODEBENCH_PREFIX,
        prompt_suffix=CODE_BENCHMARK_SUFFIX,
        post_process=postprocess_bigcodebench
    ),
    'HumanEval': BenchmarkConfig(
        schema=CodeOutput,
        prompt_prefix=CODE_BENCHMARK_PREFIX,
        prompt_suffix=CODE_BENCHMARK_SUFFIX,
        post_process=postprocess_humaneval
    ),
    'GAIA': BenchmarkConfig(
        schema=TextOutput,
        prompt_prefix=QA_BENCHMARK_PREFIX,
        prompt_suffix='',
        post_process=postprocess_qa
    ),
    'PaperBench': BenchmarkConfig(
        schema=None,
        prompt_prefix='',
        prompt_suffix='',
        post_process=postprocess_qa
    ),
    'SWEBench': BenchmarkConfig(
        schema=CodeOutput,
        prompt_prefix=CODE_BENCHMARK_PREFIX,
        prompt_suffix=CODE_BENCHMARK_SUFFIX,
        post_process=postprocess_humaneval
    ),
    'WritingBench': BenchmarkConfig(
        schema=TextOutput,
        prompt_prefix=QA_BENCHMARK_PREFIX,
        prompt_suffix='',
        post_process=postprocess_qa
    ),
}

DEFAULT_CONFIG = BenchmarkConfig(schema=None)


def get_benchmark_config(benchmark_type: str) -> BenchmarkConfig:
    return BENCHMARK_REGISTRY.get(benchmark_type, DEFAULT_CONFIG)


def get_schema_for_benchmark(benchmark_type: str) -> Optional[type[BaseModel]]:
    return get_benchmark_config(benchmark_type).schema


def enhance_prompt(prompt: str, benchmark_type: str) -> str:
    config = get_benchmark_config(benchmark_type)
    return f'{config.prompt_prefix}{prompt}{config.prompt_suffix}'


def post_process_output(output: str, benchmark_type: str) -> str:
    config = get_benchmark_config(benchmark_type)
    return config.post_process(output)
