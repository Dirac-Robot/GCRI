from enum import Enum
from typing import List, Literal
from typing import Optional

from pydantic import BaseModel, Field, model_validator
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
    intent_analysis: str = Field(
        default='',
        description='User intent summary (scope/output type). Omit if locked_intent provided.'
    )
    strictness: Literal['strict', 'moderate', 'creative'] = Field(
        ...,
        description='The strictness level inferred from the task. Must be applied to all strategies.'
    )
    strategies: List[Strategy] = Field(..., description='List of generated strategies.')


class Hypothesis(BaseModel):
    hypothesis: str = Field(
        ..., 
        description='The candidate answer or solution strategy to be evaluated and refined. It represents the core content to be tested against the task requirements.'
    )


class Reasoning(BaseModel):
    refined_hypothesis: str = Field(
        ...,
        description='A lightly improved version of the hypothesis. It should fix logical flaws and enhance clarity without changing the original solution family or exceeding the scope.'
    )
    reasoning: str = Field(
        ...,
        description='Detailed evaluation of the hypothesis, strictly identifying strengths, weaknesses, and likely failure points before refinement.'
    )


class Verification(BaseModel):
    counter_example: str = Field(
        ...,
        description='A specific scenario where the hypothesis fails. Keep it concise (max 3-5 sentences).'
    )
    counter_strength: Literal['strong', 'moderate', 'weak', 'none'] = Field(
        ...,
        description='The severity of the failure. Use "strong" for execution errors or fatal logic flaws, and "none" if the hypothesis is valid.'
    )
    adjustment: str = Field(
        ...,
        description='A concise log describing exactly what was changed in the hypothesis and why, to fix the counter-example.'
    )
    reasoning: str = Field(
        ...,
        description='Analysis of why the counter-example is valid.'
    )


class BranchAnalysis(BaseModel):
    branch_index: int = Field(..., description='0-based index of the branch being evaluated')
    summary_hypothesis: str = Field(..., description='Core idea summary of the hypothesis (1-2 sentences)')
    summary_counter_example: str = Field(..., description='Core summary of the raised counter-example')
    status: RefutationStatus = Field(..., description='Refutation status: valid (perfect), refuted (fatal flaw), or partial (incomplete)')
    failure_category: FailureCategory = Field(
        default=FailureCategory.NONE,
        description='Category of failure if not valid: logic_error, req_missing, hallucination, practicality, other'
    )
    reasoning: str = Field(..., description='Basis for judgment on whether the hypothesis survived or was rejected')


class DecisionProtoType(BaseModel):
    decision: bool = Field(..., description='True if at least one branch is valid and ready as final answer')
    best_branch_index: int = Field(
        default=-1,
        ge=-1,
        description='0-based index of the winning branch. MUST be >= 0 when decision is True.'
    )
    global_feedback: Optional[str] = Field(
        None,
        description='Strategic direction for the next turn synthesizing all failures'
    )
    branch_evaluations: List[BranchAnalysis] = Field(..., description='Detailed evaluation list for each branch')

    @model_validator(mode='after')
    def validate_branch_index(self):
        if self.decision and self.best_branch_index < 0:
            raise ValueError('best_branch_index must be >= 0 when decision is True')
        return self


class PlanProtoType(BaseModel):
    thought: str = Field(description='Reasoning for the current analysis and plan formulation')
    next_task: Optional[str] = Field(description='Specific single task to be performed next (None if finished)')
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
