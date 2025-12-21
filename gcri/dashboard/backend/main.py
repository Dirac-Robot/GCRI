from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import sys
import subprocess
from loguru import logger

from gcri.dashboard.backend.manager import manager
from gcri.config import scope
from gcri.dashboard.backend.watcher import watcher

app = FastAPI(title="GCRI Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class LogMessage(BaseModel):
    """
    Standard format for logs coming from GCRI agent via HTTP sink.
    """
    record: Dict[str, Any]

class TaskRequest(BaseModel):
    task: str

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, maybe receive commands from frontend later
            data = await websocket.receive_text()
            # Echo or process if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- API Endpoints ---
@app.post("/api/log")
async def receive_log(log: LogMessage):
    """
    Receives logs/state updates from the GCRI agent process.
    Broadcasts them to all connected frontend clients via WebSocket.
    """
    # Simply broadcast the entire log record structure
    # The frontend will parse it to update the graph/terminal
    await manager.broadcast({"type": "log", "data": log.record})
    return {"status": "ok"}

@app.post("/api/run")
async def run_task(req: TaskRequest):
    """
    Spawns a new GCRI agent process with the given task.
    """
    task_text = req.task
    if not task_text:
        return {"error": "Task cannot be empty"}

    logger.info(f"Received task request: {task_text}")
    
    # Spawn the process
    try:
        cmd = [sys.executable, '-m', 'gcri', task_text]
        subprocess.Popen(cmd, cwd=scope.config.project_dir)
        
        return {"status": "started", "message": f"Task '{task_text[:20]}...' initiated."}
    except Exception as e:
        logger.error(f"Failed to spawn task: {e}")
        return {"error": str(e)}

@app.get("/api/file")
async def get_file_content(path: str):
    """
    Returns the content of a file.
    Security: simplified for local dev, but should check if path is within monitored roots.
    """
    if not os.path.exists(path):
        return {"error": "File not found"}
    
    # Security check: ensure path is within monitored directories
    # Logic: must start with one of the monitored root paths
    allowed = False
    
    # If watcher hasn't started or has no paths, we might need a fallback check 
    # or just trust the config.
    allowed_roots = watcher.monitored_paths
    if not allowed_roots and scope.config.project_dir:
         allowed_roots = [os.path.abspath(scope.config.project_dir)]

    abs_path = os.path.abspath(path)
    for root in allowed_roots:
        if abs_path.startswith(root):
            allowed = True
            break
    
    if not allowed:
        return {"error": "Access denied"}
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        return {"error": str(e)}

# --- Startup/Shutdown ---
@app.on_event("startup")
async def startup_event():
    # Determine paths to monitor
    monitoring_paths = scope.config.dashboard.monitor_directories
    
    # Default to project directory if not specified
    if not monitoring_paths and scope.config.project_dir:
        monitoring_paths = [scope.config.project_dir]
        
    if monitoring_paths:
        watcher.start(monitoring_paths)

@app.on_event("shutdown")
async def shutdown_event():
    watcher.stop()

# --- Static Files ---
# Mount frontend build directory.
frontend_dist_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")

if os.path.exists(frontend_dist_path):
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="frontend")
else:
    # Fallback/Dev message if built files are missing
    @app.get("/")
    def read_root():
        return {"message": "GCRI Dashboard Backend is running. Frontend build not found."}
