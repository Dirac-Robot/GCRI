from typing import List

from pydantic import BaseModel


class Strategies(BaseModel):
    strategies: List[str]


class Hypothesis(BaseModel):
    hypothesis: str


class Reasoning(BaseModel):
    refined_hypothesis: str
    reasoning: str


class Verification(BaseModel):
    counter_example: str
    counter_strength: str
    adjustment: str
    reasoning: str


class Decision(BaseModel):
    decision: bool
    final_output: str
    feedback: str
    decision_reasoning: str


class Compression(BaseModel):
    compressed_output: str
