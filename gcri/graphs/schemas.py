from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

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


class ConstraintList(BaseModel):
    new_active_constraints: List[str] = Field(
        description='Extract all explicit, mandatory rules or constraints from the input feedback. '
                    'Must be clear, independent sentences.'
    )


class Strategies(BaseModel):
    strategies: List[str]


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


class Compression(BaseModel):
    compressed_output: str


class BranchAnalysis(BaseModel):
    branch_index: int
    summary_hypothesis: str = Field(..., description='가설의 핵심 아이디어 요약 (1-2문장)')
    summary_counter_example: str = Field(..., description='제기된 반례의 핵심 내용 요약')
    status: RefutationStatus = Field(..., description='반례에 의한 가설의 기각 여부')
    failure_category: FailureCategory
    reasoning: str = Field(..., description='왜 이 가설이 생존/기각 되었는지에 대한 판단 근거')


class Decision(BaseModel):
    decision: bool = Field(..., description='최종 승인 여부 (하나라도 완벽하면 True)')
    final_output: Optional[str] = Field(None, description='최종 채택된 완벽한 답변')
    global_feedback: Optional[str] = Field(None, description='모든 실패를 종합했을 때 다음 턴에 필요한 전략적 방향성')
    branch_evaluations: List[BranchAnalysis] = Field(..., description='각 브랜치에 대한 상세 평가 리스트')
