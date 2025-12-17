import operator
from typing import List, Annotated, Optional, Literal

from pydantic import BaseModel, Field

from gcri.graphs.schemas import BranchAnalysis, RefutationStatus, Strategy


class IterationLog(BaseModel):
    count_in_memory_log: int
    global_feedback: str
    branch_evaluations: List[BranchAnalysis]

    def get_summary_line(self):
        summaries = []
        for branch in self.branch_evaluations:
            if branch.status != RefutationStatus.VALID:
                summaries.append(f'(!) Strategy "{branch.summary_hypothesis}" failed due to {branch.failure_category}')
        return '\n'.join(summaries)


class StructuredMemory(BaseModel):
    active_constraints: List[str] = Field(default_factory=list)
    history: List[IterationLog] = Field(default_factory=list)

    def format_for_strategy(self, template):
        constraints = '\n'.join([f'- {c}' for c in self.active_constraints])
        graveyard = ''
        for log in self.history:
            summary = log.get_summary_line()
            if summary:
                graveyard += f'- [Iter {log.count_in_memory_log}] {summary}\n'
        last_log = self.history[-1] if self.history else None
        if last_log:
            recent = f'Global Policy for Next: {last_log.global_feedback}\n'
            for br in last_log.branch_evaluations:
                recent += f'   * Branch {br.branch_index} Error: {br.reasoning}\n'
        else:
            recent = ''
        return template.format(constraints=constraints, graveyard=graveyard, recent=recent)


class HypothesisResult(BaseModel):
    index: int
    strategy: str
    hypothesis: str
    reasoning: str
    counter_reasoning: str
    counter_example: str
    counter_strength: str
    adjustment: str


class TaskState(BaseModel):
    count: int = 0
    task: str
    task_strictness: Literal['strict', 'moderate', 'creative'] = 'moderate'
    strategies: List[Strategy] = Field(
        default_factory=list,
        description='List of generated strategies.'
    )
    results: Annotated[List[HypothesisResult], operator.add] = Field(default_factory=list)
    best_branch_index: Optional[int] = None
    aggregated_result: Optional[str] = None
    decision: Optional[bool] = None
    final_output: Optional[str] = None
    global_feedback: Optional[str] = None
    branch_evaluations: List[BranchAnalysis] = Field(default_factory=list)
    memory: StructuredMemory = Field(default_factory=StructuredMemory)
    feedback: str = ''


class BranchState(BaseModel):
    task_in_branch: str
    count_in_branch: int = 0
    strictness: Literal['strict', 'moderate', 'creative'] = 'moderate'
    strategy: Strategy
    index: int
    hypothesis: Optional[str] = None
    reasoning: Optional[str] = None
    work_dir: str
    results: List[HypothesisResult] = Field(default_factory=list)


class GlobalState(BaseModel):
    goal: str
    knowledge_context: List[str] = Field(default_factory=list)
    current_task: Optional[str] = None
    final_answer: Optional[str] = None
    mid_result: Optional[str] = None
    plan_count: int = 0
    memory: StructuredMemory = Field(default_factory=StructuredMemory)

