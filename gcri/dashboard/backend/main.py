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
from gcri.dashboard.backend.web_callbacks import WebCallbacks
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
    commit_mode: str = 'manual'  # 'auto-accept', 'auto-reject', 'manual'


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


class WorkspaceRequest(BaseModel):
    work_dir: str


@app.post('/api/workspace/files')
async def get_workspace_files(request: WorkspaceRequest):
    """Get list of files in a branch workspace directory."""
    work_dir = request.work_dir
    if not work_dir or not os.path.exists(work_dir):
        return {'files': [], 'error': 'Directory not found'}

    files = []
    ignore_patterns = {'.git', '__pycache__', 'venv', 'env', 'node_modules', '.idea', '.vscode'}
    try:
        for root, dirs, filenames in os.walk(work_dir):
            dirs[:] = [d for d in dirs if d not in ignore_patterns and not d.startswith('.')]
            rel_dir = os.path.relpath(root, work_dir)
            if rel_dir == '.':
                rel_dir = ''
            for f in filenames:
                if f.startswith('.'):
                    continue
                files.append({
                    'name': f,
                    'path': os.path.join(rel_dir, f) if rel_dir else f,
                    'full_path': os.path.join(root, f)
                })
    except Exception as e:
        return {'files': [], 'error': str(e)}

    return {'files': files}


class FileContentRequest(BaseModel):
    file_path: str


@app.post('/api/workspace/file')
async def get_file_content(request: FileContentRequest):
    """Get content of a specific file."""
    file_path = request.file_path
    if not file_path or not os.path.exists(file_path):
        return {'content': '', 'error': 'File not found'}

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return {'content': content}
    except Exception as e:
        return {'content': '', 'error': str(e)}


@app.get('/api/status')
async def get_status():
    return {
        'running': execution_lock.locked(),
        'aborted': abort_event.is_set()
    }


@app.post('/api/abort')
async def abort_task():
    global current_task_thread
    if not execution_lock.locked() and current_task_thread is None:
        return {'status': 'no_task', 'message': 'No task is currently running.'}

    logger.warning('🛑 Abort requested by user!')
    abort_event.set()

    # Cancel the asyncio task if it exists
    if current_task_thread is not None and not current_task_thread.done():
        current_task_thread.cancel()
        logger.info('Task cancellation requested.')

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


class CommitResponse(BaseModel):
    approved: bool


@app.post('/api/commit/respond')
async def respond_to_commit(response: CommitResponse):
    """Receive commit approval/rejection from frontend."""
    from gcri.dashboard.backend.web_callbacks import WebCallbacks

    callbacks = WebCallbacks.get_instance()
    if not callbacks:
        return {'status': 'error', 'message': 'No active callbacks instance.'}

    if not callbacks.is_pending_commit():
        return {'status': 'error', 'message': 'No pending commit request.'}

    callbacks.receive_commit_response(response.approved)
    action = 'approved' if response.approved else 'rejected'
    logger.info(f'Commit {action} by user.')
    return {'status': 'ok', 'action': action}


@app.get('/api/commit/status')
async def get_commit_status():
    """Check if there's a pending commit request."""
    from gcri.dashboard.backend.web_callbacks import WebCallbacks

    callbacks = WebCallbacks.get_instance()
    if not callbacks:
        return {'pending': False}

    return {'pending': callbacks.is_pending_commit()}

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

    # Set commit mode on WebCallbacks before execution
    from gcri.dashboard.backend.web_callbacks import WebCallbacks
    callbacks = WebCallbacks.get_instance()
    if callbacks:
        callbacks.set_commit_mode(task_request.commit_mode)

    # Broadcast state reset event to frontend
    asyncio.run_coroutine_threadsafe(
        manager.broadcast({
            'type': 'log',
            'data': {
                'record': {
                    'level': {'name': 'INFO'},
                    'message': 'Starting new task...',
                    'extra': {
                        'ui_event': 'state_reset'
                    }
                }
            }
        }),
        main_event_loop
    )

    logger.info(f'🚀 Integrated Runner received task: {task_request.task} (Mode: {task_request.agent_mode}, Commit: {task_request.commit_mode})')

    async def _execute_async():
        global current_task_thread
        with execution_lock:
            try:
                # Run the blocking agent call in a thread pool
                await run_in_threadpool(target_agent, task_request.task)
            except asyncio.CancelledError:
                logger.warning('🛑 Task cancelled by user.')
                raise
            except Exception as e:
                if abort_event.is_set():
                    logger.warning('🛑 Task aborted by user.')
                else:
                    logger.error(f'Execution failed: {e}')
            finally:
                current_task_thread = None

    # Store the task so we can cancel it
    current_task_thread = asyncio.create_task(_execute_async())

    return {'status': 'started', 'message': 'Task execution started in background.'}


@app.on_event('startup')
async def startup_event():
    global gcri_instance, main_event_loop
    try:
        main_event_loop = asyncio.get_running_loop()
        
        logger.info('⚙️ Initializing Unified GCRI Backend...')

        # Create web callbacks with the event loop
        web_callbacks = WebCallbacks(event_loop=main_event_loop)

        # Load BOTH agents
        gcri_instance = {}

        try:
            gcri_instance['unit'] = GCRI(scope.config, abort_event=abort_event, callbacks=web_callbacks)
            logger.info('✅ GCRI Unit Instance Ready.')
        except Exception as e:
            logger.error(f'❌ Failed to load Unit Agent: {e}')

        try:
            gcri_instance['planner'] = GCRIMetaPlanner(scope.config, abort_event=abort_event, callbacks=web_callbacks)
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
