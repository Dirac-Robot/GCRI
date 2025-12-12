gcri
====

한 줄 요약
---------
GCRI는 LLM 기반의 다중-브랜치 가설-검증 루프(생성→정제→검증→결정)를 실행하는 경량 파이프라인으로, 단일 작업을 여러 전략으로 탐색해 최종 결정을 도출하거나 메타-플래너를 통해 여러 작업을 단계적으로 수행합니다.

빠른 시작 (설치 & 실행)
-----------------------
사전 요구사항
- Python 3.11+ 권장
- pip로 dependencies 설치 (requirements.txt 포함)
- .env 파일에 모델/키 관련 환경변수 설정(예: OPENAI_API_KEY 등)

1) 가상환경 생성 및 의존성 설치

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2) 기본 실행 (대화형)

- GCRI 단위 워커 실행 (한 작업을 여러 전략으로 처리)

```bash
python -m gcri.entry
# 또는
python -c "from gcri.entry import cli_entry; cli_entry()"
```
터미널에 나타나는 프롬프트에 분석하고 싶은 작업(또는 작업을 적어둔 파일 경로)을 입력하세요.

- 메타-플래너 모드 (계획 → 여러 GCRI 작업 위임)

```bash
python -m gcri.entry plan
```

3) 프로그램 방식 사용 (간단 예)

```python
from gcri.config import scope
from gcri.graphs.gcri_unit import GCRI

config = scope()  # config 객체를 반환
unit = GCRI(config)
result = unit('문장 단위로 목표를 입력하세요')
print(result['final_output'])
```

핵심 예제 (가장 흔한 사용 사례)
-----------------------------
- 목표: 복잡한 문제(예: 설계/정책/디버깅 과제)를 여러 추론 전략으로 병렬 탐색하여 신뢰 가능한 결론을 얻고자 할 때.
- 실행: python -m gcri.entry → 작업 입력 → 내부적으로 여러 브랜치(기본 3개)가 생성되어 가설 생성→정제→검증→결정 과정을 순환.

아키텍처 개요
---------------
주요 구성 요소
- gcri.config
  - Scope 기반 설정 관리 및 템플릿/에이전트 파라미터 선언.
- gcri.entry
  - CLI 진입점. 'plan' 인자 유무에 따라 단위 워커 또는 메타-플래너를 실행.
- gcri.main / gcri.main_planner
  - 각각 단위 GCRI 실행 루프와 메타-플래너 실행 루프를 다룸.
- gcri.graphs.gcri_unit.GCRI
  - 핵심 워크플로우: 전략 샘플링 → 브랜치별 가설 생성/정제/검증 → 집계 → 결정 → 메모리 업데이트.
  - 주요 공개 메서드: __call__(task, initial_memory=None) → iteration 결과(결정/출력/메모리)
- gcri.graphs.planner.GCRIMetaPlanner
  - 상위 플래너: 목표(goal)를 받아 여러 단일 작업으로 분해하고 각 작업을 GCRI unit에 위임.
  - 공개 메서드: __call__(goal) → 최종 상태(혹은 실패 로그)
- gcri.tools.cli
  - 로컬 도구(토이) 래퍼: execute_shell_command, read_file, write_file, local_python_interpreter
  - build_model(model_id, gcri_options=None, **parameters) : 내부적으로 CodeAgentBuilder를 반환하여 에이전트(LLM) 객체 생성.
- 템플릿 (gcri/templates/*.txt)
  - 각 단계(hypothesis, reasoning, verification, strategy_generator, decision, memory 등)에 사용되는 프롬프트 템플릿.

데이터/스키마
- gcri.graphs.schemas : Pydantic 모델을 사용해 에이전트 출력(가설/검증/결정/플랜/압축 등)을 정의.
- gcri.graphs.states : 워크플로우 상태(Iteration, Memory, TaskState 등)를 pydantic으로 표현.

공개 API 표면 (요약)
- Classes / 함수
  - gcri.graphs.gcri_unit.GCRI
    - 사용법: GCRI(config)(task, initial_memory=None)
    - 반환: dict(contains keys like 'decision','final_output','memory', ...)
  - gcri.graphs.planner.GCRIMetaPlanner
    - 사용법: GCRIMetaPlanner(config)(goal)
    - 반환: dict containing 'final_answer' or planner logs.
  - gcri.tools.cli.build_model(model_id, gcri_options=None, **parameters)
    - LLM 에이전트 빌더. gcri_options.use_code_tools에 따라 로컬 툴 접근 허용.
  - gcri.tools.cli.execute_shell_command / read_file / write_file / local_python_interpreter
    - 로컬 환경에서의 보조 실행 도구(InteractiveToolGuard를 통해 실행 제어).
  - gcri.entry.cli_entry()
    - CLI 스크립트 엔트리포인트(엔트리로 사용 권장).

템플릿/설정
- 템플릿은 프로젝트 동작의 핵심(프롬프트 형태). 기본 경로는 gcri/templates/*.txt.
- 설정은 gcri/config.py의 Scope를 통해 변경 가능. 모델 아이디, 파라미터, 브랜치 수 등 조정.

문서 자동생성 지침
------------------
자동 도구 사용 권장 (Python 코드 기준)
- Sphinx + autodoc (권장)
  1) 설치: pip install sphinx sphinx-autodoc-typehints
  2) 초기화: sphinx-quickstart docs -q
  3) conf.py: extensions=['sphinx.ext.autodoc','sphinx.ext.napoleon','sphinx_autodoc_typehints']
  4) autodoc 예: docs/index.rst에서 .. automodule:: gcri.graphs.gcri_unit
  5) 빌드: sphinx-build -b html docs/ docs/_build/html
- 대체: pdoc 또는 mkdocstrings (Markdown-oriented) 사용 가능.

자동 문서 생성이 불가능한 경우(또는 빠른 배포용) — 최소 수동 API 참조 포함
- README의 "공개 API 표면" 섹션을 최신 상태로 유지.
- 각 주요 클래스(GCRI, GCRIMetaPlanner, build_model)에 사용 예제와 입력/출력 예시를 포함.
- TODO: gcri/README_API.md 를 생성하여 함수별 시그니처와 간단 예시를 추가.

테스트·빌드·기여
-----------------
테스트
- 현재 프로젝트 루트에 pytest 관련 출력 파일(pytest_output.txt)이 존재하지만 별도 테스트 스위트가 보이지 않습니다.
- 권장: pytest 기반 테스트 디렉토리 tests/ 추가 후 실행

```bash
pip install pytest
pytest -q
```

정적 검사
- flake8 / pylint 사용 권장. 예시:

```bash
pip install pylint
pylint gcri
```

빌드
- 순수 Python 패키지로 배포하려면 pyproject.toml이 있습니다. 패키징/배포는 poetry 또는 pip build 사용.

기여 가이드 (간단)
- Fork → feature 브랜치 → PR
- 코드 스타일: black + isort 권장
- 커밋 메시지: 명확한 변경 목적 기입
- 새 템플릿/모델 설정을 추가할 때는 templates 디렉토리와 config 업데이트를 잊지 말 것

라이선스
- 현재 루트에 LICENSE 파일이 없습니다. 오픈 소스 공개를 원한다면 MIT/Apache-2.0 등 적절한 라이선스를 추가하세요.

트러블슈팅 (FAQ)
------------------
1) 에이전트 초기화 실패(LLM 모델 관련)
   - .env에 필요한 인증 키가 있는지 확인하세요.
   - gcri/config.py에서 모델 id와 파라미터가 올바른지 확인.

2) 템플릿 파일을 찾을 수 없음
   - config.templates 경로를 확인하세요. 상대경로로 지정되어 있으므로 작업 디렉토리에 따라 경로가 달라질 수 있습니다.

3) 도구 실행이 터미널에서 멈춤 — InteractiveToolGuard 프롬프트
   - 로컬 도구(execute_shell_command 등)는 interactive guard에 의해 실행 전 사용자 확인을 요구합니다. 자동 실행을 원하면 InteractiveToolGuard 내부의 auto_mode 로직을 수정하거나, 도구 사용을 비활성화하세요 (gcri_options.use_code_tools=False).

4) 로그/출력이 저장되지 않음
   - 기본 로그 디렉토리는 gcri/config.py의 config.templates.log_dir에 의해 결정됩니다. 쓰기 권한과 경로 존재 여부를 확인하세요.

문서화 TODO (권장)
- 각 템플릿(gcri/templates/*.txt)에 간단한 주석(입력 필드 설명)을 추가.
- gcri/tools/cli.PythonREPL, InteractiveToolGuard 동작 문서화.
- README_API.md 생성: 함수/클래스/모듈별 예제와 반환값 샘플 JSON 추가.

마지막 참고사항
----------------
- 이 프로젝트는 LLM(대화형 모델)을 워크플로우 엔진(langgraph)과 결합하여 여러 추론 경로를 병렬·반복적으로 테스트하고 결정하는 파이프라인입니다. 템플릿(프롬프트) 설계이 품질을 결정하므로 실제 사용에서는 템플릿 수정과 config 튜닝이 필수입니다.

Contact
-------
- 리포지토리 유지관리자 또는 코드 기여자는 README를 통해 연락처/이슈 템플릿을 추가하세요.
