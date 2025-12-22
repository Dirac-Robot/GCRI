from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import asyncio
import threading
from loguru import logger
import pathlib

from gcri.dashboard.backend.manager import manager
from gcri.config import scope
from gcri.dashboard.backend.watcher import watcher
from gcri.graphs.gcri_unit import GCRI
from gcri.graphs.planner import GCRIMetaPlanner


app = FastAPI(title='GCRI Dashboard')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

gcri_instance: Optional[Any] = None  # GCRI or GCRIMetaPlanner
execution_lock = threading.Lock()
main_event_loop = None
abort_event = threading.Event()
current_task_thread: Optional[threading.Thread] = None


class LogMessage(BaseModel):
    record: Dict[str, Any]


class TaskRequest(BaseModel):
    task: str
    agent_mode: str = 'unit'  # 'unit' or 'planner'


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def websocket_sink(message):
    try:
        import json
        record = json.loads(message)

        if main_event_loop and main_event_loop.is_running():
             asyncio.run_coroutine_threadsafe(
                 manager.broadcast({'type': 'log', 'data': record.get('record', record)}),
                 main_event_loop
             )
    except Exception as e:
        pass


@app.post('/api/log')
async def receive_log(log: LogMessage):
    await manager.broadcast({'type': 'log', 'data': log.record})
    return {'status': 'ok'}


@app.get('/api/status')
async def get_status():
    return {
        'running': execution_lock.locked(),
        'aborted': abort_event.is_set()
    }


@app.post('/api/abort')
async def abort_task():
    global current_task_thread
    if not execution_lock.locked():
        return {'status': 'no_task', 'message': 'No task is currently running.'}

    logger.warning('🛑 Abort requested by user!')
    abort_event.set()

    # Broadcast abort event to frontend
    if main_event_loop and main_event_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({'type': 'log', 'data': {
                'record': {
                    'level': {'name': 'WARNING'},
                    'message': '🛑 Task aborted by user.',
                    'extra': {'ui_event': 'abort'}
                }
            }}),
            main_event_loop
        )

    return {'status': 'aborting', 'message': 'Abort signal sent. Task will terminate shortly.'}


@app.post('/api/run')
async def run_task(task_request: TaskRequest):
    global gcri_instance, current_task_thread

    if not gcri_instance:
         # Should not happen as we init attempts to load
         return {'error': 'GCRI system not initialized.'}

    target_agent = None
    if task_request.agent_mode == 'planner':
        if isinstance(gcri_instance, dict) and 'planner' in gcri_instance:
            target_agent = gcri_instance['planner']
        else:
            return {'error': 'Planner agent not available.'}
    else:
        if isinstance(gcri_instance, dict) and 'unit' in gcri_instance:
            target_agent = gcri_instance['unit']
        else:
            return {'error': 'Unit agent not available.'}

    if not task_request.task:
        return {'error': 'Task cannot be empty'}

    if execution_lock.locked():
        return {'error': 'Another task is currently running. Please wait.'}

    # Reset abort event for new task
    abort_event.clear()

    logger.info(f'🚀 Integrated Runner received task: {task_request.task} (Mode: {task_request.agent_mode})')

    def _execute():
        with execution_lock:
             try:
                 # Standardize call
                 target_agent(task_request.task)
             except KeyboardInterrupt:
                 logger.warning('Task interrupted.')
             except Exception as e:
                 if abort_event.is_set():
                     logger.warning('Task aborted by user.')
                 else:
                     logger.error(f'Execution failed: {e}')

    asyncio.create_task(run_in_threadpool(_execute))

    return {'status': 'started', 'message': 'Task execution started in background.'}


@app.on_event('startup')
async def startup_event():
    global gcri_instance, main_event_loop
    try:
        main_event_loop = asyncio.get_running_loop()
        
        logger.info('⚙️ Initializing Unified GCRI Backend...')
        
        # Load BOTH agents
        gcri_instance = {}
        
        try:
            gcri_instance['unit'] = GCRI(scope.config)
            logger.info('✅ GCRI Unit Instance Ready.')
        except Exception as e:
            logger.error(f'❌ Failed to load Unit Agent: {e}')

        try:
            gcri_instance['planner'] = GCRIMetaPlanner(scope.config)
            logger.info('✅ GCRIMetaPlanner Instance Ready.')
        except Exception as e:
             logger.error(f'❌ Failed to load Planner Agent: {e}')

        logger.add(websocket_sink, serialize=True, level='INFO')
        logger.info('✅ WebSocket Logger Sink Attached.')

        monitoring_paths = scope.config.dashboard.monitor_directories
        if not monitoring_paths and scope.config.project_dir:
            monitoring_paths = [scope.config.project_dir]
        if monitoring_paths:
            watcher.start(monitoring_paths)
            
    except Exception as e:
        logger.error(f'❌ Failed to initialize backend components: {e}')


@app.on_event('shutdown')
async def shutdown_event():
    watcher.stop()


backend_dir = pathlib.Path(__file__).parent.resolve()
frontend_dist_path = backend_dir.parent / 'frontend' / 'dist'

logger.info(f'Frontend Dist Path: {frontend_dist_path}')

if frontend_dist_path.exists():
    assets_path = frontend_dist_path / 'assets'
    if assets_path.exists():
        logger.info(f'Mounting assets from: {assets_path}')
        app.mount('/assets', StaticFiles(directory=str(assets_path)), name='assets')
    
    logger.info(f'Mounting root static files from: {frontend_dist_path}')
    app.mount('/', StaticFiles(directory=str(frontend_dist_path), html=True), name='frontend')
else:
    logger.warning(f'Frontend build directory not found at: {frontend_dist_path}')
    @app.get('/')
    def read_root():
        return {'message': 'GCRI Dashboard Backend is running. Frontend build not found.'}
