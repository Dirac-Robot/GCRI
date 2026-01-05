import asyncio
import os
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

load_dotenv()

from gcri.config import scope
from gcri.graphs.gcri_unit import GCRI
from gcri.tools.docker_sandbox import get_sandbox
from gcri.benchmark.schemas import get_schema_for_benchmark, CodeOutput, enhance_prompt, post_process_output


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


_config = None
_solve_lock: Optional[asyncio.Lock] = None


def get_solve_lock() -> asyncio.Lock:
    global _solve_lock
    if _solve_lock is None:
        _solve_lock = asyncio.Lock()
    return _solve_lock


def create_gcri_instance(schema=None) -> GCRI:
    global _config
    if _config is None:
        _config = scope.config
    gcri = GCRI(_config, schema=schema)
    logger.info(f'🤖 GCRI Benchmark Instance Created (schema={schema.__name__ if schema else None})')
    return gcri


def set_config(config):
    global _config
    _config = config


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('🚀 GCRI Benchmark Server Starting...')
    yield
    logger.info('🛑 GCRI Benchmark Server Shutting Down...')
    if _config:
        try:
            get_sandbox(_config).clean_up_all()
        except Exception as e:
            logger.warning(f'Failed to clean up sandbox: {e}')


app = FastAPI(title='GCRI Benchmark Server', lifespan=lifespan)


@app.post('/benchmark/solve')
async def solve(request: BenchmarkRequest) -> BenchmarkResponse:
    logger.info('='*60)
    logger.info(f'📥 START: {request.task_id}')
    benchmark_type = request.metadata.get('benchmark_type', 'Benchmark')
    schema = get_schema_for_benchmark(benchmark_type)
    async with get_solve_lock():
        try:
            gcri = create_gcri_instance(schema=schema)
            loop = asyncio.get_event_loop()
            enhanced_prompt = enhance_prompt(request.prompt, benchmark_type)
            result = await loop.run_in_executor(
                None,
                lambda: gcri(enhanced_prompt, commit_mode='auto-accept')
            )
            if result is None:
                logger.warning(f'📤 END: {request.task_id} (FAILED - None result)')
                return BenchmarkResponse(
                    task_id=request.task_id,
                    answer='',
                    success=False,
                    error='GCRI returned None (model call failed or max iterations reached)'
                )
            final_output = result.get('final_output', '')
            if final_output is None:
                final_output = ''
            elif isinstance(final_output, dict):
                if schema and issubclass(schema, CodeOutput):
                    final_output = final_output.get('code', str(final_output))
                else:
                    final_output = final_output.get('answer', str(final_output))
            final_output = post_process_output(str(final_output), benchmark_type)
            logger.info(f'📤 END: {request.task_id} (SUCCESS)')
            logger.debug(f'Output preview: {final_output[:200]}...' if len(final_output)>200 else f'Output: {final_output}')
            return BenchmarkResponse(
                task_id=request.task_id,
                answer=final_output,
                reasoning=result.get('feedback', ''),
                success=True
            )
        except Exception as e:
            logger.error(f'📤 END: {request.task_id} (ERROR: {e})')
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
    config.protocols.max_iterations = 3
    config.protocols.force_output = True


@scope
def run_server(config):
    import uvicorn
    set_config(config)
    port = config.benchmark.get('server_port', 8001)
    if not isinstance(port, int):
        port = 8001
    logger.info(f'🚀 Starting GCRI Benchmark Server on port {port}')
    uvicorn.run(app, host='0.0.0.0', port=port)


if __name__ == '__main__':
    run_server()
