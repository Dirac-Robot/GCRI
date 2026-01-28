---
name: Callback Pattern for Environment Adaptation
description: Extensible callback interface for adapting behavior across CLI, Web, API environments
---

# Callback Pattern for Environment Adaptation

환경(CLI, 웹, API, 테스트)에 따라 동작을 다르게 구현하기 위한 콜백 인터페이스 패턴.

## 핵심 구조

```python
from typing import Any, Dict, Optional

class BaseCallbacks:
    """
    기본 콜백 인터페이스. 모든 메서드에 합리적인 기본값(no-op 또는 auto-approve)을 제공.
    환경별로 오버라이드하여 사용.
    """

    def on_commit_request(self, context: Dict[str, Any]) -> bool:
        """변경사항 반영 전 승인 요청"""
        return True  # Default: auto-approve

    def on_node_update(self, node: str, branch: Optional[int], data: Dict[str, Any]):
        """노드 상태 변경 시 호출, UI 업데이트나 로깅용"""
        pass

    def on_phase_change(self, phase: str):
        """실행 단계 변경 시 호출"""
        pass

    def on_iteration_complete(self, iteration: int, result: Dict[str, Any]):
        """반복 완료 시 호출"""
        pass
```

## 환경별 구현체

```python
# 1. CLI 환경 - 사용자 입력 대기
class CLICallbacks(BaseCallbacks):
    def on_commit_request(self, context: Dict[str, Any]) -> bool:
        try:
            response = input('변경사항을 적용하시겠습니까? (y/n): ')
            return response.lower().strip() == 'y'
        except (EOFError, KeyboardInterrupt):
            return False

# 2. 자동 승인 - 벤치마크/테스트용
class AutoCallbacks(BaseCallbacks):
    def on_commit_request(self, context: Dict[str, Any]) -> bool:
        return True

# 3. 모든 커밋 거부 - dry-run 테스트용
class NoCommitCallbacks(BaseCallbacks):
    def on_commit_request(self, context: Dict[str, Any]) -> bool:
        return False

# 4. 웹 환경 - WebSocket 이벤트 전송
class WebCallbacks(BaseCallbacks):
    def __init__(self, websocket):
        self.ws = websocket
    
    def on_node_update(self, node: str, branch: Optional[int], data: Dict[str, Any]):
        import asyncio
        asyncio.create_task(self.ws.send_json({
            'type': 'node_update',
            'node': node,
            'branch': branch,
            'data': data
        }))
```

## 사용 패턴

```python
class Executor:
    def __init__(self, config, callbacks=None):
        # 콜백이 없으면 기본값(AutoCallbacks) 사용
        self.callbacks = callbacks or AutoCallbacks()
    
    def execute(self):
        # ... 작업 수행 ...
        
        # 커밋 필요 시 콜백 호출
        context = {'file_path': '/path', 'changes': changes}
        if self.callbacks.on_commit_request(context):
            self.apply_changes()
        else:
            self.discard_changes()
        
        # 상태 업데이트 브로드캐스트
        self.callbacks.on_node_update('step1', branch=0, data={'status': 'complete'})
```

## 사용 시점

- 동일 로직을 CLI/웹/API 등 다양한 환경에서 실행할 때
- 테스트 시 사용자 입력 없이 자동 진행이 필요할 때
- 실행 중 상태 변경을 외부에 알려야 할 때

## 참고 파일

- `gcri/graphs/callbacks.py` - GCRICallbacks 기본 인터페이스 및 구현체들
