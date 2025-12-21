import asyncio
import os
import time
from typing import List, Dict, Any
from threading import Thread
from loguru import logger
from gcri.dashboard.backend.manager import manager

class WorkspaceWatcher:
    """
    Monitors specified directories for file changes and broadcasts the file tree to connected clients.
    Uses polling to avoid complex dependency management (watchdog) for now.
    """
    def __init__(self, poll_interval: float = 2.0):
        self.poll_interval = poll_interval
        self.monitored_paths: List[str] = []
        self._stop_event = False
        self._thread: Thread = None
        self._last_state: Dict[str, float] = {}  # Map path -> mtime

    def start(self, paths: List[str]):
        """Starts the monitoring thread."""
        self.monitored_paths = [os.path.abspath(p) for p in paths if os.path.exists(p)]
        if not self.monitored_paths:
            logger.warning("No valid paths to monitor.")
            return

        logger.info(f"Starting WorkspaceWatcher for: {self.monitored_paths}")
        self._stop_event = False
        self._thread = Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the monitoring thread."""
        self._stop_event = True
        if self._thread:
            self._thread.join()

    def _monitor_loop(self):
        """Background loop to check for file changes."""
        while not self._stop_event:
            try:
                current_state = self._scan_files()
                if self._has_changes(current_state):
                    self._last_state = current_state
                    # Broadcast new tree
                    tree = self._build_file_tree()
                    asyncio.run(manager.broadcast({"type": "workspace_update", "data": tree}))
            except Exception as e:
                logger.error(f"Error in WorkspaceWatcher: {e}")
            
            time.sleep(self.poll_interval)

    def _scan_files(self) -> Dict[str, float]:
        """Scans all monitored paths and returns a dict of file_path -> mtime."""
        state = {}
        for root_path in self.monitored_paths:
            for dirpath, _, filenames in os.walk(root_path):
                # Skip hidden directories like .git
                if '/.' in dirpath:
                    continue
                    
                for f in filenames:
                    if f.startswith('.'): continue # Skip hidden files
                    
                    full_path = os.path.join(dirpath, f)
                    try:
                        mtime = os.path.getmtime(full_path)
                        state[full_path] = mtime
                    except OSError:
                        pass
        return state

    def _has_changes(self, current_state: Dict[str, float]) -> bool:
        """Compares current state with last state."""
        if set(current_state.keys()) != set(self._last_state.keys()):
            return True # File added or removed
        
        for path, mtime in current_state.items():
            if path not in self._last_state or self._last_state[path] != mtime:
                return True # Modified
        
        return False

    def _build_file_tree(self) -> List[Dict[str, Any]]:
        """Builds a nested dictionary representation of the file system for the frontend."""
        # For simplicity, we'll return a flat list of files with metadata first, 
        # allowing frontend to build the tree or just show a list.
        # Actually, let's return a structure grouped by root path.
        
        tree = []
        for root_path in self.monitored_paths:
            root_name = os.path.basename(root_path)
            files = []
            for dirpath, _, filenames in os.walk(root_path):
                if '/.' in dirpath: continue
                
                rel_dir = os.path.relpath(dirpath, root_path)
                if rel_dir == '.': rel_dir = ""
                
                for f in filenames:
                    if f.startswith('.'): continue
                    files.append({
                        "name": f,
                        "path": os.path.join(rel_dir, f),
                        "full_path": os.path.join(dirpath, f),
                        "type": "file"
                    })
            
            tree.append({
                "name": root_name,
                "path": root_path,
                "type": "directory",
                "children": files
            })
            
        return tree

watcher = WorkspaceWatcher()
