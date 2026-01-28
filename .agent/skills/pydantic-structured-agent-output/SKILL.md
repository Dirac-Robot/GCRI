---
name: Pydantic Structured Agent Output
description: Pattern for defining and enforcing structured LLM output schemas using Pydantic
---

# Pydantic Structured Agent Output Pattern

LLM 에이전트의 출력을 Pydantic 스키마로 구조화하여 안정적인 결과를 보장하는 패턴.

## 기본 스키마 정의

```python
from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional
from enum import Enum

# 1. Enum으로 제한된 선택지 정의
class FailureCategory(str, Enum):
    NONE = 'none'
    LOGIC_ERROR = 'logic_error'
    HALLUCINATION = 'hallucination'

# 2. 필드 설명을 LLM 프롬프트로 활용
class Strategy(BaseModel):
    name: str = Field(..., description='전략의 짧은 이름')
    description: str = Field(..., description='추론 경로와 방법론에 대한 상세 설명')
    hints: List[str] = Field(
        ..., 
        description='실행 에이전트를 위한 구체적 지시사항. 구현 가능해야 함.'
    )

# 3. Literal로 정해진 옵션 강제
class Verification(BaseModel):
    counter_strength: Literal['strong', 'moderate', 'weak', 'none'] = Field(
        ...,
        description='실패의 심각도. "strong"은 치명적, "none"은 유효함을 의미.'
    )
    reasoning: str = Field(..., description='반례가 유효한 이유 분석')
```

## 2-Phase 아키텍처 스키마

GCRI의 Hypothesis → Aggregation → Verification 2단계 구조를 지원하는 스키마:

```python
# Phase 1: Hypothesis 생성 결과
class RawHypothesis(BaseModel):
    """BranchesGenerator에서 생성한 초기 가설 (검증 전)"""
    index: int = Field(..., description='원본 브랜치 인덱스')
    strategy_name: str = Field(..., description='사용된 전략 이름')
    strategy_description: str = Field(..., description='전략 설명')
    hypothesis: str = Field(..., description='생성된 가설')
    reasoning: str = Field(..., description='추론 과정')
    container_id: str = Field(..., description='샌드박스 컨테이너 ID')

# Phase 2: Aggregation 결과
class AggregatedBranch(BaseModel):
    """Aggregating Agent가 출력하는 병합된 브랜치"""
    index: int = Field(..., description='새 브랜치 인덱스')
    combined_hypothesis: str = Field(..., description='병합/정제된 가설')
    source_indices: List[int] = Field(..., description='병합에 사용된 원본 브랜치 인덱스들')
    merge_reasoning: str = Field(..., description='왜 이 조합이 선택되었는지')

class AggregationResult(BaseModel):
    """Aggregating Agent의 출력 스키마"""
    branches: List[AggregatedBranch]
    discarded_indices: List[int] = Field(default_factory=list, description='버려진 브랜치 인덱스')
    aggregation_summary: str = Field(..., description='집계 전략 요약')
```

## 동적 스키마 생성

```python
from pydantic import create_model

def create_decision_schema(custom_output_schema=None):
    """사용자 정의 출력 스키마를 포함한 Decision 스키마 동적 생성"""
    if custom_output_schema is None:
        custom_output_schema = str
    
    return create_model(
        'Decision',
        __base__=DecisionProtoType,  # 기본 필드를 가진 베이스 클래스
        final_output=(
            Optional[custom_output_schema],
            Field(None, description='요구된 스키마에 맞는 최종 구조화된 답변')
        )
    )
```

## 모델 연동 (with_structured_output)

```python
from langchain.chat_models import init_chat_model

# 1. 모델 초기화
model = init_chat_model('gpt-5-mini', max_tokens=25600)

# 2. 구조화된 출력 스키마 적용
structured_model = model.with_structured_output(schema=Strategy)

# 3. 프롬프트 전달 및 구조화된 결과 수신
result: Strategy = structured_model.invoke(template)
print(result.name)  # 타입 안전 접근
print(result.hints)
```

## Aggregator Agent 연동

```python
# AggregationResult 스키마로 구조화된 출력
aggregator_model = model.with_structured_output(schema=AggregationResult)

result: AggregationResult = aggregator_model.invoke(aggregator_template)
for branch in result.branches:
    print(f'Branch {branch.index}: {branch.combined_hypothesis}')
    print(f'  Sources: {branch.source_indices}')
```

## 검증 로직 추가

```python
class DecisionProtoType(BaseModel):
    decision: bool = Field(..., description='적어도 하나의 브랜치가 유효하면 True')
    best_branch_index: int = Field(default=-1, ge=-1)
    
    @model_validator(mode='after')
    def validate_branch_index(self):
        """decision이 True면 반드시 유효한 브랜치 인덱스 필요"""
        if self.decision and self.best_branch_index < 0:
            raise ValueError('decision이 True일 때 best_branch_index는 0 이상이어야 함')
        return self
```

## Retry 패턴과 조합

```python
def _invoke_with_retry(agent, template, error_context='agent', max_tries=3):
    """스키마 검증 실패 시 재시도"""
    for attempt in range(max_tries):
        try:
            result = agent.invoke(template)
            if result is not None:
                return result
        except ValidationError as e:
            logger.warning(f'{error_context} 스키마 검증 실패 (시도 {attempt+1}/{max_tries}): {e}')
    raise ValueError(f'{error_context}가 {max_tries}회 시도 후 출력 생성 실패')
```

## 사용 시점

- LLM 출력을 코드에서 안전하게 파싱해야 할 때
- 여러 LLM 호출 결과를 일관된 형식으로 집계할 때
- 출력 형식 오류를 자동 재시도로 복구하고 싶을 때
- Aggregator가 여러 가설을 병합할 때 구조화된 출력이 필요할 때

## 참고 파일

- `gcri/graphs/schemas.py` - 모든 Pydantic 스키마 정의 (RawHypothesis, AggregatedBranch, AggregationResult 포함)
- `gcri/graphs/gcri_unit.py` - `_invoke_with_retry`, `with_structured_output` 사용 예
- `gcri/graphs/aggregator.py` - HypothesisAggregator에서 AggregationResult 사용
