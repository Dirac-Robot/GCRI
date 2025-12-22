from enum import Enum
from typing import List, Literal
from typing import Optional, Any

from pydantic import BaseModel, Field
from pydantic import create_model


class RefutationStatus(str, Enum):
    VALID = 'valid'
    REFUTED = 'refuted'
    PARTIALLY_VALID = 'partial'


class FailureCategory(str, Enum):
    NONE = 'none'
    LOGIC_ERROR = 'logic_error'
    REQUIREMENT_MISSING = 'req_missing'
    HALLUCINATION = 'hallucination'
    PRACTICALITY_ISSUE = 'practicality'
    OTHER = 'other'


class ActiveConstraints(BaseModel):
    new_active_constraints: List[str] = Field(
        description='Extract all explicit, mandatory rules or constraints from the input feedback. '
                    'Must be clear, independent sentences.'
    )


class Strategy(BaseModel):
    name: str = Field(..., description='A short, descriptive name for this strategy.')
    description: str = Field(..., description='Detailed explanation of the reasoning path and methodology.')
    feedback_reflection: str = Field(
        ...,
        description='Summary of past failures and how this strategy specifically '
                    'addresses them (modifies reasoning path).'
    )
    hints: List[str] = Field(
        ...,
        description='Explicit directives/hints for the Hypothesis Agent. Must be implementable.'
    )


class Strategies(BaseModel):
    intent_analysis: Optional[str] = Field(
        None,
        description=(
            'Summarize the user\'s desired scope (e.g., "User wants a review, not code"). '
            'Analyze implicit Scope/Output Type. '
            'If locked_intent is provided, this may be omitted.'
        )
    )
    strictness: Literal['strict', 'moderate', 'creative'] = Field(
        ...,
        description='The strictness level inferred from the task. Must be applied to all strategies.'
    )
    strategies: List[Strategy] = Field(..., description='List of generated strategies.')


class Hypothesis(BaseModel):
    hypothesis: str


class Reasoning(BaseModel):
    refined_hypothesis: str
    reasoning: str


class Verification(BaseModel):
    counter_example: str = Field(
        ...,
        description='A specific scenario where the hypothesis fails. Keep it concise (max 3-5 sentences).'
    )
    counter_strength: Literal['strong', 'moderate', 'weak', 'none']
    adjustment: str
    reasoning: str = Field(
        ...,
        description='Analysis of why the counter-example is valid.'
    )


class BranchAnalysis(BaseModel):
    branch_index: int
    summary_hypothesis: str = Field(..., description='Core idea summary of the hypothesis (1-2 sentences)')
    summary_counter_example: str = Field(..., description='Core summary of the raised counter-example')
    status: RefutationStatus = Field(..., description='Refutation status of the hypothesis by the counter-example')
    failure_category: FailureCategory
    reasoning: str = Field(..., description='Basis for judgment on whether the hypothesis survived or was rejected')


class DecisionProtoType(BaseModel):
    decision: bool = Field(..., description='Final approval decision (True if at least one is perfect)')
    best_branch_index: Optional[int] = None
    # final_output: Optional[Any] = Field(None, description='The final adopted perfect answer')
    global_feedback: Optional[str] = Field(
        None,
        description='Strategic direction for the next turn synthesizing all failures'
    )
    branch_evaluations: List[BranchAnalysis] = Field(..., description='Detailed evaluation list for each branch')


class PlanProtoType(BaseModel):
    thought: str = Field(description='Reasoning for the current analysis and plan formulation')
    next_task: Optional[str] = Field(description='Specific single task to be performed next (None if finished)')
    # final_answer: Optional[Any] = Field(description='Final answer if the goal is achieved (None if in progress)')
    is_finished: bool = Field(description='Whether the goal has been achieved')


class Compression(BaseModel):
    summary: str = Field(description='Updated high-level knowledge summary incorporating the latest result.')
    retained_constraints: List[str] = Field(
        description='Filtered list of active constraints. Remove duplicates, obsolete rules, or trivial details.'
    )
    discard_reason: str = Field(description='Brief reason why certain details were compressed or discarded.')


def create_decision_schema(schema=None):
    if schema is None:
        schema = str
    return create_model(
        'Decision',
        __base__=DecisionProtoType,
        final_output=(
            Optional[schema],
            Field(None, description='The final structured answer matching the required schema.')
        )
    )


def create_planner_schema(schema=None):
    if schema is None:
        schema = str
    return create_model(
        'Plan',
        __base__=PlanProtoType,
        final_output=(
            Optional[schema],
            Field(None, description='Final answer matching the required schema. Fill ONLY when is_finished is True.')
        )
    )
