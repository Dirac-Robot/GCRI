import httpx
from typing import Any
from inspect_ai.model import ModelAPI, ModelOutput, GenerateConfig, modelapi
from inspect_ai._eval.context import task_run_dir


@modelapi(name='gcri')
def gcri_model_api():
    return GCRIModelAPI


BENCHMARK_TYPE_MAP = {
    'bigcodebench': 'BigCodeBench',
    'humaneval': 'HumanEval',
    'gaia': 'GAIA',
    'swe_bench': 'SWEBench',
    'paperbench': 'PaperBench',
    'writingbench': 'WritingBench',
}


class GCRIModelAPI(ModelAPI):
    def __init__(
        self,
        model_name: str = 'gcri',
        base_url: str = None,
        timeout: float = 1800.0,
        benchmark_type: str = None,
        **kwargs
    ):
        super().__init__(model_name=model_name, base_url=base_url, **kwargs)
        self.endpoint = base_url or 'http://localhost:8001'
        self.timeout = timeout
        self._benchmark_type = benchmark_type

    def _detect_benchmark_type(self) -> str:
        if self._benchmark_type:
            return self._benchmark_type
        try:
            run_dir = task_run_dir()
            if run_dir:
                run_dir_str = str(run_dir).lower()
                for key, value in BENCHMARK_TYPE_MAP.items():
                    if key in run_dir_str:
                        return value
        except Exception:
            pass
        return 'Benchmark'

    async def generate(
        self,
        input: list,
        tools: list[Any] = [],
        tool_choice: Any = None,
        config: GenerateConfig = GenerateConfig(),
        **kwargs
    ) -> ModelOutput:
        prompt_parts = []
        for msg in input:
            if hasattr(msg, 'content'):
                content = msg.content
                if isinstance(content, list):
                    content = ' '.join(str(c) for c in content)
                prompt_parts.append(str(content))
            else:
                prompt_parts.append(str(msg))
        prompt = '\n'.join(prompt_parts)

        benchmark_type = self._detect_benchmark_type()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f'{self.endpoint}/benchmark/solve',
                json={
                    'task_id': f'{benchmark_type}-model',
                    'prompt': prompt,
                    'metadata': {'benchmark_type': benchmark_type}
                }
            )
            response.raise_for_status()
            result = response.json()

        answer = result.get('answer', '')
        return ModelOutput.from_content(model=self.model_name, content=answer)
