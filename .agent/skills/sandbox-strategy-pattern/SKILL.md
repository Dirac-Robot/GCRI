---
name: Sandbox Strategy Pattern
description: Interchangeable sandbox implementations (Docker/Local) for isolated code execution
---

# Sandbox Strategy Pattern

Docker와 Local 파일시스템을 상호 교환 가능한 샌드박스 전략 패턴. 환경에 따라 격리 수준을 선택.

## 공통 인터페이스

```python
from typing import Protocol, Dict, List

class SandboxProtocol(Protocol):
    """모든 샌드박스 구현체가 따라야 할 인터페이스"""
    
    def setup_branch(self, iteration: int, branch: int, source_dir: str) -> str:
        """브랜치 환경 생성, 고유 ID 반환"""
        ...
    
    def setup_verification_branches(
        self,
        iteration: int,
        aggregated_branches: List['AggregatedBranch'],
        source_containers: Dict[int, str]
    ) -> Dict[int, str]:
        """집계된 브랜치를 위한 검증 컨테이너 매핑 생성"""
        ...
    
    def execute_command(self, container_id: str, command: str) -> str:
        """셸 명령 실행"""
        ...
    
    def execute_python(self, container_id: str, code: str) -> str:
        """파이썬 코드 실행"""
        ...
    
    def commit_to_host(self, container_id: str, target_dir: str) -> None:
        """결과물을 호스트로 복사"""
        ...
    
    def clean_up_container(self, container_id: str) -> None:
        """개별 컨테이너/디렉토리 정리"""
        ...
    
    def clean_up_all(self) -> None:
        """모든 리소스 정리"""
        ...
    
    def get_container(self, iteration: int, branch: int) -> str:
        """브랜치 ID로 컨테이너/디렉토리 조회"""
        ...
```

## Verification 브랜치 매핑 패턴

GCRI의 2-Phase 아키텍처에서는 Aggregation 후 검증 컨테이너를 재매핑:

```python
def setup_verification_branches(
    self,
    iteration: int,
    aggregated_branches: List[AggregatedBranch],
    source_containers: Dict[int, str]
) -> Dict[int, str]:
    """
    집계된 브랜치를 위한 검증 컨테이너 생성
    
    전략:
    1. 단일 소스인 경우: 기존 컨테이너 재사용
    2. 다중 소스 병합인 경우: 새 컨테이너 생성 후 파일 병합
    
    Args:
        iteration: 현재 iteration 인덱스
        aggregated_branches: Aggregator가 생성한 AggregatedBranch 리스트
        source_containers: 원본 브랜치 인덱스 → 컨테이너 ID 매핑
    
    Returns:
        검증 브랜치 인덱스 → 컨테이너 ID 매핑
    """
    verification_containers = {}
    
    for branch in aggregated_branches:
        if len(branch.source_indices) == 1:
            # 단일 소스 → 기존 컨테이너 재사용
            src_idx = branch.source_indices[0]
            verification_containers[branch.index] = source_containers[src_idx]
        else:
            # 다중 소스 → 새 컨테이너 생성
            new_container = self._create_merged_container(
                iteration,
                branch.index,
                [source_containers[i] for i in branch.source_indices]
            )
            verification_containers[branch.index] = new_container
    
    return verification_containers
```

## Docker 구현체

```python
class DockerSandbox:
    def __init__(self, config):
        self.image = config.sandbox.image
        self.timeout = config.sandbox.timeout
        self.memory_limit = config.sandbox.memory_limit
        self.cpu_limit = config.sandbox.cpu_limit
        self.network_mode = config.sandbox.network_mode
        self._containers = {}
    
    def setup_branch(self, iteration: int, branch: int, source_dir: str) -> str:
        container_name = f'gcri_branch_{iteration}_{branch}_{uuid.uuid4().hex[:8]}'
        create_cmd = [
            'docker', 'create',
            '--name', container_name,
            f'--memory={self.memory_limit}',
            f'--cpus={self.cpu_limit}',
            f'--network={self.network_mode}',
            '-w', '/workspace',
            '-t', self.image,
            'tail', '-f', '/dev/null'
        ]
        # 컨테이너 생성 및 파일 복사
        result = subprocess.run(create_cmd, capture_output=True, text=True)
        container_id = result.stdout.strip()
        subprocess.run(['docker', 'start', container_id])
        self._copy_to_container(source_dir, container_id)
        self._containers[f'{iteration}_{branch}'] = container_id
        return container_id
    
    def execute_command(self, container_id: str, command: str) -> str:
        exec_cmd = ['docker', 'exec', container_id, 'sh', '-c', command]
        result = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=self.timeout)
        return result.stdout + result.stderr
    
    def _create_merged_container(
        self,
        iteration: int,
        branch_index: int,
        source_containers: List[str]
    ) -> str:
        """여러 컨테이너의 파일을 병합한 새 컨테이너 생성"""
        # TODO: 파일 병합 로직 구현
        # 현재는 첫 번째 소스 컨테이너 사용
        return source_containers[0]
```

## Local 구현체

```python
class LocalSandbox:
    """Docker 없이 로컬 파일시스템에서 격리 실행"""
    
    def __init__(self, config):
        self.timeout = config.sandbox.timeout
        self._branches = {}
    
    def setup_branch(self, iteration: int, branch: int, source_dir: str) -> str:
        branch_dir = f'/tmp/gcri_local_{iteration}_{branch}_{uuid.uuid4().hex[:8]}'
        ignore_patterns = {'.git', '__pycache__', 'venv', 'node_modules', '.gcri'}
        
        os.makedirs(branch_dir, exist_ok=True)
        for item in os.listdir(source_dir):
            if item in ignore_patterns:
                continue
            src = os.path.join(source_dir, item)
            dst = os.path.join(branch_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*ignore_patterns))
            else:
                shutil.copy2(src, dst)
        
        self._branches[f'{iteration}_{branch}'] = branch_dir
        return branch_dir
    
    def execute_command(self, branch_dir: str, command: str) -> str:
        result = subprocess.run(command, shell=True, cwd=branch_dir,
                                capture_output=True, text=True, timeout=self.timeout)
        return result.stdout + result.stderr
```

## 전략 선택 팩토리

```python
def get_sandbox(config):
    """config에 따라 적절한 샌드박스 반환"""
    if config.sandbox.get('enabled', True):
        sandbox = DockerSandbox(config)
        if sandbox.docker_available:
            return sandbox
        logger.warning('Docker unavailable, falling back to LocalSandbox')
    return LocalSandbox(config)
```

## 컨테이너 재사용 전략

| 시나리오 | 처리 방식 |
|----------|-----------|
| Hypothesis 생성 | 각 브랜치별 새 컨테이너 생성 |
| 단일 소스 Verification | 기존 컨테이너 재사용 (동일 작업공간 유지) |
| 다중 소스 Verification | 첫 번째 소스 컨테이너 사용 (TODO: 파일 병합) |

## 사용 시점

- 코드 실행을 격리된 환경에서 수행해야 할 때
- Docker-in-Docker 이슈를 피해야 할 때 (InspectAI 벤치마크 등)
- 테스트 환경에서 빠른 실행이 필요할 때
- Aggregation 후 브랜치 재매핑이 필요할 때

## 참고 파일

- `gcri/tools/docker_sandbox.py` - DockerSandbox 구현
- `gcri/tools/local_sandbox.py` - LocalSandbox 구현
- `gcri/tools/utils.py` - SandboxManager 및 `setup_verification_branches` 메서드
- `gcri/graphs/schemas.py` - AggregatedBranch 스키마
