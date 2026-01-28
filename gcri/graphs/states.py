import operator
from typing import List, Annotated, Optional, Literal, Dict, Any

from pydantic import BaseModel, Field, model_validator

from gcri.graphs.schemas import (
    BranchAnalysis, RefutationStatus, Strategy,
    RawHypothesis, AggregatedBranch
)


class IterationLog(BaseModel):
    count_in_memory_log: int = Field(..., description='Iteration index for this log entry')
    global_feedback: str = Field(default='', description='Strategic direction synthesized from all branch failures')
    branch_evaluations: List[BranchAnalysis] = Field(default_factory=list, description='Detailed evaluation for each branch')

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
    active_constraints: List[str] = Field(
        default_factory=list,
        description='Mandatory rules extracted from feedback. Must not be violated.'
    )
    history: List[IterationLog] = Field(
        default_factory=list,
        description='Past iteration logs with failures and learnings'
    )

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
    index: int = Field(..., description='Branch index (0-based)')
    strategy: Strategy = Field(..., description='Strategy used for this branch')
    hypothesis: str = Field(..., description='The refined hypothesis after reasoning')
    reasoning: str = Field(..., description='Evaluation of strengths, weaknesses, and failure points')
    counter_reasoning: str = Field(..., description='Analysis of why the counter-example is valid')
    counter_example: str = Field(..., description='Specific scenario where hypothesis fails')
    counter_strength: str = Field(..., description='Severity: strong (fatal), moderate, weak, or none')
    adjustment: str = Field(..., description='Concise log of what was changed to fix the counter-example')


class TaskState(BaseModel):
    """
    Global state for a single GCRI task execution.

    Tracks the current iteration, generated strategies, branch results,
    collective decision outcomes, and accumulated memory/feedback.
    """
    count: int = Field(default=0, description='Current iteration index (0-based)')
    task: str = Field(..., description='Original task description')
    intent_analysis: str = Field(default='', description='Locked user intent (scope/output type)')
    task_strictness: Literal['strict', 'moderate', 'creative'] = Field(
        default='moderate',
        description='Strictness level: strict (rigorous), moderate (balanced), creative (insight-focused)'
    )
    strategies: List[Strategy] = Field(default_factory=list, description='Generated strategies for current iteration')
    results: Annotated[List[HypothesisResult], operator.add] = Field(
        default_factory=list,
        description='Aggregated hypothesis results from all branches'
    )
    best_branch_index: int = Field(default=-1, description='Index of winning branch (-1 if none selected)')
    aggregated_result: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description='Filtered results for decision agent'
    )
    decision: Optional[bool] = Field(default=None, description='Final decision: True if accepted, False if rejected')
    final_output: Optional[Any] = Field(default=None, description='Final answer if decision is True')
    global_feedback: Optional[str] = Field(
        default=None,
        description='Strategic direction for next iteration if decision is False'
    )
    branch_evaluations: List[BranchAnalysis] = Field(
        default_factory=list,
        description='Detailed analysis of each branch'
    )
    memory: StructuredMemory = Field(default_factory=StructuredMemory, description='Accumulated learnings across iterations')
    feedback: str = Field(default='', description='Formatted feedback incorporating memory and constraints')
    # New fields for BranchesGenerator architecture
    raw_hypotheses: Annotated[List[RawHypothesis], operator.add] = Field(
        default_factory=list,
        description='Hypotheses from BranchesGenerator before aggregation'
    )
    aggregated_branches: List[AggregatedBranch] = Field(
        default_factory=list,
        description='Branches after aggregation by HypothesisAggregator'
    )
    verification_container_map: Dict[int, str] = Field(
        default_factory=dict,
        description='Mapping from verification branch index to container ID'
    )


class BranchState(BaseModel):
    """
    State for a single reasoning branch within an iteration.

    Contains the branch-specific strategy, hypothesis under development,
    and isolated Docker container for execution.
    """
    task_in_branch: str = Field(..., description='Task description for this branch')
    intent_analysis_in_branch: str = Field(default='', description='Locked user intent for this branch')
    count_in_branch: int = Field(default=0, description='Current iteration index')
    strictness: Literal['strict', 'moderate', 'creative'] = Field(
        default='moderate',
        description='Strictness level for this branch'
    )
    strategy: Strategy = Field(..., description='Strategy assigned to this branch')
    index: int = Field(..., description='Branch index (0-based)')
    hypothesis: Optional[str] = Field(default=None, description='Current hypothesis being developed')
    reasoning: Optional[str] = Field(default=None, description='Reasoning evaluation of the hypothesis')
    container_id: str = Field(..., description='Docker container ID for isolated execution')
    results: List[HypothesisResult] = Field(default_factory=list, description='Results from this branch')
    # Output field for BranchesGenerator
    raw_hypothesis_output: Optional[RawHypothesis] = Field(
        default=None,
        description='RawHypothesis output from this branch (for aggregation)'
    )


class VerificationBranchState(BaseModel):
    """
    State for a verification branch after aggregation.

    Contains the aggregated hypothesis to verify and the container for execution.
    """
    task_in_branch: str = Field(..., description='Task description for this branch')
    intent_analysis_in_branch: str = Field(default='', description='Locked user intent for this branch')
    count_in_branch: int = Field(default=0, description='Current iteration index')
    strictness: Literal['strict', 'moderate', 'creative'] = Field(
        default='moderate',
        description='Strictness level for this branch'
    )
    aggregated_branch: AggregatedBranch = Field(..., description='Aggregated branch to verify')
    index: int = Field(..., description='Verification branch index (0-based)')
    container_id: str = Field(..., description='Container ID for isolated execution')
    results: List[HypothesisResult] = Field(default_factory=list, description='Verification results')


class GlobalState(BaseModel):
    """
    State for the meta-planner across multiple GCRI task executions.

    Maintains the overall goal, accumulated knowledge context from
    completed tasks, and current planning progress.
    """
    goal: str = Field(..., description='Overall goal for multi-step planning')
    knowledge_context: List[str] = Field(
        default_factory=list,
        description='Accumulated knowledge summaries from completed tasks'
    )
    current_task: Optional[str] = Field(default=None, description='Current sub-task being executed')
    final_answer: Optional[Any] = Field(default=None, description='Final answer if goal achieved')
    mid_result: Optional[Any] = Field(default=None, description='Intermediate result from last task')
    plan_count: int = Field(default=0, description='Number of planning iterations completed')
    memory: StructuredMemory = Field(default_factory=StructuredMemory, description='Accumulated learnings')

