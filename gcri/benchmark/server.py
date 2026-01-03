
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()
from loguru import logger

from gcri.config import scope
from gcri.graphs.gcri_unit import GCRI


class BenchmarkRequest(BaseModel):
    task_id: str
    prompt: str
    metadata: dict = {}


class BenchmarkResponse(BaseModel):
    task_id: str
    answer: str
    reasoning: str = ''
    success: bool = True
    error: Optional[str] = None


_gcri_instance: Optional[GCRI] = None
_config = None


def get_gcri_instance() -> GCRI:
    global _gcri_instance, _config
    if _gcri_instance is None:
        if _config is None:
            _config = scope.config
        _gcri_instance = GCRI(_config)
        logger.info('🤖 GCRI Benchmark Instance Initialized')
    return _gcri_instance


def set_config(config):
    global _config
    _config = config


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('🚀 GCRI Benchmark Server Starting...')
    yield
    logger.info('🛑 GCRI Benchmark Server Shutting Down...')


app = FastAPI(title='GCRI Benchmark Server', lifespan=lifespan)


@app.post('/benchmark/solve')
async def solve(request: BenchmarkRequest) -> BenchmarkResponse:
    try:
        gcri = get_gcri_instance()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: gcri(request.prompt, commit_mode='auto-accept')
        )
        answer = result.get('final_output', '')
        if answer is None:
            answer = ''
        return BenchmarkResponse(
            task_id=request.task_id,
            answer=str(answer),
            reasoning=result.get('feedback', ''),
            success=True
        )
    except Exception as e:
        logger.error(f'Benchmark solve error: {e}')
        return BenchmarkResponse(
            task_id=request.task_id,
            answer='',
            success=False,
            error=str(e)
        )


@app.get('/health')
async def health():
    return {'status': 'ok'}


@scope.observe()
def benchmark_mode(config):
    config.benchmark.enabled = True
    config.benchmark.server_port = 8001
    config.benchmark.timeout = 120
    config.protocols.max_iterations = 3
    config.protocols.force_output = True


@scope
def run_server(config):
    import uvicorn
    set_config(config)
    try:
        port = config.benchmark.server_port
        if not isinstance(port, int):
            logger.warning(f'Warning: server_port is {type(port)}, using default 8001')
            port = 8001
    except Exception as e:
        logger.warning(f'Failed to get server_port: {e}, using default 8001')
        port = 8001
    logger.info(f'🚀 Starting GCRI Benchmark Server on port {port}')
    uvicorn.run(app, host='0.0.0.0', port=port)


if __name__ == '__main__':
    run_server()
