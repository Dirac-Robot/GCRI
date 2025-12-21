from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os

from gcri.dashboard.backend.manager import manager

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

# --- Static Files ---
# Mount frontend build directory.
# In development, we might just run Vite separately, but for the final CLI integration,
# we expect the frontend to be built into dashboard/frontend/dist
frontend_dist_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")

if os.path.exists(frontend_dist_path):
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="frontend")
else:
    # Fallback/Dev message if built files are missing
    @app.get("/")
    def read_root():
        return {"message": "GCRI Dashboard Backend is running. Frontend build not found."}
