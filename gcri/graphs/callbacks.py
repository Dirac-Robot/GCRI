from typing import Any, Dict, Optional


class GCRICallbacks:
    """
    Base callback interface for GCRI.
    Override methods to customize behavior for different environments (CLI, Web, API, etc.)
    All methods have sensible defaults (no-op or auto-approve).
    """

    def on_commit_request(self, context: Dict[str, Any]) -> bool:
        """
        Called when GCRI wants to commit winning branch to project root.
        
        Args:
            context: Dict containing:
                - winning_branch_path: str
                - best_branch_index: int
                - final_output: Any
                
        Returns:
            True to commit, False to discard changes.
        """
        return True  # Default: auto-approve

    def on_node_update(self, node: str, branch: Optional[int], data: Dict[str, Any]):
        """
        Called when a node's state is updated.
        Useful for UI updates or logging.
        
        Args:
            node: Node name (e.g., 'hypothesis', 'reasoning', 'verification')
            branch: Branch index (None for global nodes like 'strategy', 'decision')
            data: Node-specific data
        """
        pass

    def on_phase_change(self, phase: str):
        """
        Called when execution phase changes.
        
        Args:
            phase: Phase name ('strategy', 'execution', 'decision', 'memory', 'idle')
        """
        pass

    def on_iteration_complete(self, iteration: int, result: Dict[str, Any]):
        """
        Called when an iteration completes.
        
        Args:
            iteration: Iteration index (0-based)
            result: Iteration result containing decision, feedback, etc.
        """
        pass


class CLICallbacks(GCRICallbacks):
    """Callbacks for CLI/terminal usage with interactive prompts."""

    def on_commit_request(self, context: Dict[str, Any]) -> bool:
        try:
            response = input('Apply this result to project root? (y/n): ')
            return response.lower().strip() == 'y'
        except (EOFError, KeyboardInterrupt):
            return False


class AutoCallbacks(GCRICallbacks):
    """Callbacks that auto-approve everything. Useful for benchmarks/testing."""

    def on_commit_request(self, context: Dict[str, Any]) -> bool:
        return True


class NoCommitCallbacks(GCRICallbacks):
    """Callbacks that reject all commits. Useful for dry-run testing."""

    def on_commit_request(self, context: Dict[str, Any]) -> bool:
        return False
