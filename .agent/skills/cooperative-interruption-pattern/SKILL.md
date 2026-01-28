---
name: Cooperative Interruption Pattern
description: Pattern for graceful task abortion using threading.Event for multi-agent systems
---

# Cooperative Interruption Pattern

멀티 에이전트 시스템에서 `threading.Event`를 활용한 협력적 중단 패턴. 안전한 리소스 정리 보장.

## 핵심 구조

```python
import threading

class TaskAbortedError(Exception):
    """사용자에 의해 태스크가 중단됨을 나타내는 예외"""
    pass

class Executor:
    def __init__(self, config, abort_event=None):
        self.config = config
        self.abort_event = abort_event  # 외부에서 주입
    
    def _check_abort(self):
        """중단 요청 확인 - 각 단계 시작 전 호출"""
        if self.abort_event is not None and self.abort_event.is_set():
            logger.warning('🛑 중단 감지. 실행 중지.')
            raise TaskAbortedError('사용자에 의해 태스크 중단됨.')
    
    def execute_step(self, state):
        self._check_abort()  # 매 단계 시작 전 확인
        # ... 실제 작업 수행 ...
        return result
```

## 실행 루프에서 처리

```python
def __call__(self, task):
    self.resource_manager.setup()
    result = None
    
    try:
        for iteration in range(self.config.max_iterations):
            try:
                result = self.workflow.invoke({...})
                
                if result['decision']:
                    self._handle_success(result)
                    break
                    
            except TaskAbortedError:
                logger.warning(f'Iteration {iteration+1} 사용자에 의해 중단됨.')
                raise  # 상위로 전파
                
            except Exception as e:
                logger.error(f'Iteration {iteration+1} 오류: {e}')
                
    except (KeyboardInterrupt, TaskAbortedError) as e:
        is_abort = isinstance(e, TaskAbortedError)
        logger.warning('🛑 태스크 중단됨.' if is_abort else 'Ctrl+C로 중단됨.')
        
        if result:
            result['final_output'] = '사용자에 의해 중단됨.'
        else:
            result = {'final_output': '첫 반복 완료 전 중단됨.'}
            
    finally:
        # 항상 리소스 정리 - 중단되더라도
        self.resource_manager.clean_up()
        logger.info('🧹 리소스 정리 완료.')
    
    return result
```

## 외부에서 중단 트리거

```python
# 멀티스레드 환경에서 외부 컨트롤러가 중단 트리거
class TaskController:
    def __init__(self):
        self.abort_event = threading.Event()
        self.executor = Executor(config, abort_event=self.abort_event)
    
    def run_in_background(self, task):
        def _run():
            try:
                self.result = self.executor(task)
            except Exception as e:
                self.error = e
        
        self.thread = threading.Thread(target=_run)
        self.thread.start()
    
    def abort(self):
        """외부에서 호출하여 실행 중단"""
        self.abort_event.set()
        if self.thread:
            self.thread.join(timeout=10)  # 정리 시간 대기
```

## 웹 환경 (FastAPI + WebSocket)

```python
from fastapi import WebSocket

class TaskSession:
    def __init__(self):
        self.abort_event = threading.Event()
    
    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        
        # 태스크 시작
        task_thread = threading.Thread(
            target=lambda: self.run_task(websocket)
        )
        task_thread.start()
        
        # 클라이언트 메시지 대기
        while True:
            message = await websocket.receive_json()
            if message.get('type') == 'abort':
                self.abort_event.set()
                await websocket.send_json({'status': 'abort_requested'})
                break
```

## 중단 지점 선택

중단 확인은 다음 지점에서 수행:

1. **각 노드 시작 전** - 가장 빈번, 빠른 응답
2. **반복 시작 전** - 자연스러운 중단점
3. **외부 API 호출 전** - 불필요한 비용 방지
4. **파일 쓰기 전** - 부분 상태 방지

```python
def sample_hypothesis(self, state):
    self._check_abort()  # 노드 진입점에서 확인
    # ... 작업 수행 ...

def verify(self, state):
    self._check_abort()  # 각 노드마다 적용
    # ... 작업 수행 ...
```

## 사용 시점

- 장시간 실행되는 멀티 에이전트 워크플로우
- 웹 UI에서 사용자가 실행 중 취소할 수 있어야 할 때
- 리소스(Docker 컨테이너, 임시 파일) 정리가 필수인 경우

## 참고 파일

- `gcri/graphs/gcri_unit.py` - `_check_abort`, `TaskAbortedError` 사용
- `gcri/dashboard/backend/main.py` - abort_event 기반 세션 관리
