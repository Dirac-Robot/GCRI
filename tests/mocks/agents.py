import json
from typing import Any, List, Optional, Type
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable


class ScenarioBasedMockLLM(Runnable):
    def __init__(self, scenarios: List[dict], default_response: str = 'Mock Default'):
        self.scenarios = scenarios
        self.default_response = default_response
        self._call_log = []

    def invoke(self, input: Any, config: Optional[Any] = None) -> Any:
        prompt = ''
        if isinstance(input, str):
            prompt = input
        elif isinstance(input, list) and isinstance(input[0], BaseMessage):
            prompt = '\n'.join([m.content for m in input])
        elif isinstance(input, dict):
            prompt = str(input)
        self._call_log.append(prompt)

        for case in self.scenarios:
            trigger = case.get('trigger')
            if trigger and trigger in prompt:
                return case.get('response')
        return self.default_response

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema: Type[BaseModel]):
        return StructuredMockWrapper(self, schema)


class StructuredMockWrapper(Runnable):
    def __init__(self, llm, schema):
        self.llm = llm
        self.schema = schema

    def invoke(self, input: Any, config: Optional[Any] = None) -> Any:
        response = self.llm.invoke(input, config)
        if isinstance(response, BaseModel):
            return response
        if isinstance(response, dict):
            return self.schema(**response)
        try:
            data = json.loads(response)
            return self.schema(**data)
        except (json.JSONDecodeError, TypeError):
            return response
