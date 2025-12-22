from typing import List, Dict, Any
from fastapi import WebSocket
from loguru import logger


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.log_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f'Client connected. Total active: {len(self.active_connections)}')
        # Send existing log history to new client
        if self.log_history:
            try:
                await websocket.send_json({'type': 'history', 'data': self.log_history})
            except Exception as e:
                logger.error(f'Failed to send history to new client: {e}')

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f'Client disconnected. Total active: {len(self.active_connections)}')

    async def broadcast(self, message: Dict[str, Any]):
        # Store log messages in history
        if message.get('type') == 'log':
            self.log_history.append(message['data'])
            if len(self.log_history) > self.max_history_size:
                self.log_history = self.log_history[-self.max_history_size:]
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f'Failed to send message to client: {e}')
                pass


manager = ConnectionManager()

