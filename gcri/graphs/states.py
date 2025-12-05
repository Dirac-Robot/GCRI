import operator
from typing import List, Annotated, Optional

from pydantic import BaseModel, Field


class HypothesisResult(BaseModel):
    index: int
    hypothesis: str
    reasoning: str
    counter_reasoning: str
    counter_example: str
    counter_strength: str
    adjustment: str


class TaskState(BaseModel):
    task: str
    feedback: str = ''
    strategies: List[str] = Field(default_factory=list)
    results: Annotated[List[HypothesisResult], operator.add] = Field(default_factory=list)
    aggregated_result: Optional[str] = None
    decision: Optional[bool] = None
    final_output: Optional[str] = None
    compressed_output: Optional[str] = None
    decision_reasoning: Optional[str] = None


class BranchState(BaseModel):
    original_task: str
    strategy: str
    index: int
    hypothesis: Optional[str] = None
    reasoning: Optional[str] = None
    results: List[HypothesisResult] = Field(default_factory=list)
