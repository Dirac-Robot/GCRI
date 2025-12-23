import operator
from typing import List, Annotated, Optional, Literal, Dict, Any

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
    """
    Persistent memory structure that accumulates learnings across iterations.

    Stores active constraints (rules to follow) and iteration history
    (past decisions and their outcomes) to guide future reasoning.
    """
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
    strategy: Strategy
    hypothesis: str
    reasoning: str
    counter_reasoning: str
    counter_example: str
    counter_strength: str
    adjustment: str


class TaskState(BaseModel):
    """
    Global state for a single GCRI task execution.

    Tracks the current iteration, generated strategies, branch results,
    collective decision outcomes, and accumulated memory/feedback.
    """
    count: int = 0
    task: str
    intent_analysis: str = ''
    task_strictness: Literal['strict', 'moderate', 'creative'] = 'moderate'
    strategies: List[Strategy] = Field(
        default_factory=list,
        description='List of generated strategies.'
    )
    results: Annotated[List[HypothesisResult], operator.add] = Field(default_factory=list)
    best_branch_index: Optional[int] = None
    aggregated_result: Optional[List[Dict[str, Any]]] = None
    decision: Optional[bool] = None
    final_output: Optional[Any] = None
    global_feedback: Optional[str] = None
    branch_evaluations: List[BranchAnalysis] = Field(default_factory=list)
    memory: StructuredMemory = Field(default_factory=StructuredMemory)
    feedback: str = ''


class BranchState(BaseModel):
    """
    State for a single reasoning branch within an iteration.

    Contains the branch-specific strategy, hypothesis under development,
    and isolated workspace directory for file operations.
    """
    task_in_branch: str
    intent_analysis_in_branch: str = ''
    count_in_branch: int = 0
    strictness: Literal['strict', 'moderate', 'creative'] = 'moderate'
    strategy: Strategy
    index: int
    hypothesis: Optional[str] = None
    reasoning: Optional[str] = None
    work_dir: str
    results: List[HypothesisResult] = Field(default_factory=list)


class GlobalState(BaseModel):
    """
    State for the meta-planner across multiple GCRI task executions.

    Maintains the overall goal, accumulated knowledge context from
    completed tasks, and current planning progress.
    """
    goal: str
    knowledge_context: List[str] = Field(default_factory=list)
    current_task: Optional[str] = None
    final_answer: Optional[Any] = None
    mid_result: Optional[Any] = None
    plan_count: int = 0
    memory: StructuredMemory = Field(default_factory=StructuredMemory)

