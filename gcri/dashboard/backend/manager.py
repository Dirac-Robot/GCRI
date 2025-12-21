from typing import List, Dict, Any
from fastapi import WebSocket
from loguru import logger


class ConnectionManager:
    """
    Manages WebSocket connections for the GCRI Dashboard.
    Allows broadcasting specific events to all connected clients (Frontend).
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected. Total active: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcasts a JSON message to all connected clients.
        """
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                # We might want to remove broken connections here, but disconnect() usually handles it
                pass

manager = ConnectionManager()
