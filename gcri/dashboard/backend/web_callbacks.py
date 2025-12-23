import asyncio
import threading
from typing import Any, Dict, Optional

from gcri.graphs.callbacks import GCRICallbacks
from gcri.dashboard.backend.manager import manager


class WebCallbacks(GCRICallbacks):
    """
    Callbacks for web dashboard.
    Sends requests to frontend via WebSocket and waits for responses.
    """

    # Class-level instance for API access
    _instance: Optional['WebCallbacks'] = None
    COMMIT_TIMEOUT_SECONDS = 300  # 5 minutes timeout for user response

    def __init__(self, event_loop=None, commit_mode: str = 'manual'):
        """
        Initialize WebCallbacks.

        Args:
            event_loop: asyncio event loop for broadcasting.
            commit_mode: One of 'auto-accept', 'auto-reject', 'manual'.
        """
        self.event_loop = event_loop
        self.commit_mode = commit_mode
        self._commit_response: Optional[bool] = None
        self._commit_event = threading.Event()
        self._pending_commit = False
        WebCallbacks._instance = self

    def set_commit_mode(self, mode: str):
        """Set the commit mode. Should only be called before task execution."""
        self.commit_mode = mode

    @classmethod
    def get_instance(cls) -> Optional['WebCallbacks']:
        """Get the current WebCallbacks instance."""
        return cls._instance

    def _broadcast_sync(self, message: Dict[str, Any]):
        """Broadcast message to WebSocket clients from a sync context."""
        if self.event_loop and self.event_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(message),
                self.event_loop
            )

    def receive_commit_response(self, approved: bool):
        """
        Receive commit response from frontend API.
        Called by the /api/commit/respond endpoint.
        """
        self._commit_response = approved
        self._commit_event.set()

    def on_commit_request(self, context: Dict[str, Any]) -> bool:
        """
        Handle commit request based on commit_mode.

        - 'auto-accept': Immediately return True.
        - 'auto-reject': Immediately return False.
        - 'manual': Show UI and wait for user response.
        """
        # Handle auto modes
        if self.commit_mode == 'auto-accept':
            self._broadcast_sync({
                'type': 'log',
                'data': {
                    'record': {
                        'level': {'name': 'INFO'},
                        'message': f"🏆 Task completed! Auto-accepting changes (Branch #{context.get('best_branch_index', 0)+1})",
                        'extra': {'ui_event': 'commit_auto_accept'}
                    }
                }
            })
            return True

        if self.commit_mode == 'auto-reject':
            self._broadcast_sync({
                'type': 'log',
                'data': {
                    'record': {
                        'level': {'name': 'INFO'},
                        'message': f"🏆 Task completed! Auto-rejecting changes (Branch #{context.get('best_branch_index', 0)+1})",
                        'extra': {'ui_event': 'commit_auto_reject'}
                    }
                }
            })
            return False

        # Manual mode: wait for user response
        # Reset state
        self._commit_event.clear()
        self._commit_response = None
        self._pending_commit = True

        # Broadcast commit request to frontend
        self._broadcast_sync({
            'type': 'log',
            'data': {
                'record': {
                    'level': {'name': 'INFO'},
                    'message': f"🏆 Task completed! Waiting for commit approval...",
                    'extra': {
                        'ui_event': 'commit_request',
                        'context': {
                            'winning_branch_path': context.get('winning_branch_path'),
                            'best_branch_index': context.get('best_branch_index'),
                            'final_output': str(context.get('final_output', ''))[:500]  # Truncate for display
                        }
                    }
                }
            }
        })

        # Wait for user response with timeout
        received = self._commit_event.wait(timeout=self.COMMIT_TIMEOUT_SECONDS)
        self._pending_commit = False

        if not received:
            # Timeout - default to reject
            self._broadcast_sync({
                'type': 'log',
                'data': {
                    'record': {
                        'level': {'name': 'WARNING'},
                        'message': '⏰ Commit request timed out. Changes discarded.',
                        'extra': {'ui_event': 'commit_timeout'}
                    }
                }
            })
            return False

        return self._commit_response or False

    def is_pending_commit(self) -> bool:
        """Check if there's a pending commit request."""
        return self._pending_commit

    def on_node_update(self, node: str, branch: Optional[int], data: Dict[str, Any]):
        """Broadcast node updates to frontend."""
        self._broadcast_sync({
            'type': 'log',
            'data': {
                'record': {
                    'level': {'name': 'INFO'},
                    'message': f'Node update: {node}',
                    'extra': {
                        'ui_event': 'node_update',
                        'node': node,
                        'branch': branch,
                        'data': data
                    }
                }
            }
        })

    def on_phase_change(self, phase: str):
        """Broadcast phase changes to frontend."""
        self._broadcast_sync({
            'type': 'log',
            'data': {
                'record': {
                    'level': {'name': 'INFO'},
                    'message': f'Phase changed to: {phase}',
                    'extra': {
                        'ui_event': 'phase_change',
                        'phase': phase
                    }
                }
            }
        })
