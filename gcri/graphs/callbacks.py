from typing import Any, Dict, List, Optional

from loguru import logger


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
        return True

    def on_iteration_start(self, iteration: int, max_iterations: int):
        """Called at the start of each iteration."""
        pass

    def on_iteration_complete(self, iteration: int, result: Dict[str, Any]):
        """
        Called when an iteration completes.

        Args:
            iteration: Iteration index (0-based)
            result: Iteration result containing decision, feedback, branch_evaluations, etc.
        """
        pass

    def on_phase_change(self, phase: str, iteration: int = 0, **kwargs):
        """
        Called when execution phase changes.

        Args:
            phase: Phase name ('strategy', 'aggregation', 'verification', 'decision', 'memory', 'complete', 'idle')
            iteration: Current iteration index
        """
        pass

    def on_strategies_generated(self, iteration: int, strategies: List[Dict[str, Any]]):
        """Called after strategies are generated for an iteration."""
        pass

    def on_hypothesis_generated(self, iteration: int, branch: int, hypothesis: str, strategy_name: str):
        """Called after a branch generates a hypothesis."""
        pass

    def on_verification_complete(self, iteration: int, branch: int, counter_strength: str, counter_example: str):
        """Called after a branch verification completes."""
        pass

    def on_refinement_complete(self, iteration: int, branch: int, adjustment_log: str):
        """Called after a branch refinement completes in the verify-refine micro-loop."""
        pass

    def on_decision(self, iteration: int, decision: bool, best_branch: int, feedback: Optional[str], evaluations: List[Dict[str, Any]]):
        """Called after the decision phase."""
        pass

    def on_task_complete(self, result: Optional[Dict[str, Any]], elapsed_seconds: float):
        """Called when the entire GCRI task completes."""
        pass

    def on_task_error(self, error: Exception):
        """Called when the GCRI task fails with an error."""
        pass

    def on_task_abort(self, error: Exception):
        """Called when the GCRI task is aborted by user (explicit or KeyboardInterrupt)."""
        pass


class CLICallbacks(GCRICallbacks):
    """Callbacks for CLI/terminal usage with loguru logging and interactive prompts."""

    def on_commit_request(self, context: Dict[str, Any]) -> bool:
        try:
            response = input('Apply this result to project root? (y/n): ')
            return response.lower().strip() == 'y'
        except (EOFError, KeyboardInterrupt):
            return False

    def on_iteration_start(self, iteration, max_iterations):
        logger.info(f'🔄 Starting Iteration {iteration+1}/{max_iterations}')

    def on_iteration_complete(self, iteration, result):
        decision = result.get('decision', False)
        status = '✅ ACCEPTED' if decision else '🔁 CONTINUING'
        logger.info(f'Iteration {iteration+1} {status}')

    def on_phase_change(self, phase, iteration=0, **kwargs):
        icons = {
            'strategy': '🎯', 'aggregation': '🔀', 'verification': '🔍',
            'decision': '⚖️', 'memory': '🧠', 'complete': '🏁', 'idle': '💤',
        }
        icon = icons.get(phase, '📌')
        logger.info(f'{icon} Phase: {phase}')

    def on_strategies_generated(self, iteration, strategies):
        for i, strat in enumerate(strategies):
            name = strat.get('name', f'Strategy {i}')
            logger.info(f'  📋 Strategy {i+1}: {name}')

    def on_hypothesis_generated(self, iteration, branch, hypothesis, strategy_name):
        logger.info(f'  💡 Branch[{branch}] hypothesis generated (strategy: {strategy_name})')

    def on_verification_complete(self, iteration, branch, counter_strength, counter_example):
        logger.info(f'  🔍 Branch[{branch}] verification: {counter_strength.upper()} counter')

    def on_refinement_complete(self, iteration, branch, adjustment_log):
        logger.info(f'  🔧 Branch[{branch}] refined: {adjustment_log[:80]}')

    def on_decision(self, iteration, decision, best_branch, feedback, evaluations):
        if decision:
            logger.info(f'  ✅ Decision: ACCEPT (branch #{best_branch+1})')
        else:
            logger.info(f'  ❌ Decision: REJECT — {(feedback or "")[:100]}')

    def on_task_complete(self, result, elapsed_seconds):
        logger.info(f'🏁 Task completed in {elapsed_seconds:.1f}s ({elapsed_seconds/60:.1f}min)')

    def on_task_error(self, error):
        logger.error(f'💥 Task failed: {error}')


class AutoCallbacks(GCRICallbacks):
    """Callbacks that auto-approve everything. Useful for benchmarks/testing."""

    def on_commit_request(self, context: Dict[str, Any]) -> bool:
        return True


class NoCommitCallbacks(GCRICallbacks):
    """Callbacks that reject all commits. Useful for dry-run testing."""

    def on_commit_request(self, context: Dict[str, Any]) -> bool:
        return False
