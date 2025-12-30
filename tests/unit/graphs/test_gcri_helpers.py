import pytest
from unittest.mock import MagicMock
from gcri.graphs.gcri_unit import GCRI
from gcri.graphs.states import StructuredMemory


class TestGCRIHelpers:
    @pytest.fixture
    def gcri_instance(self, mock_config):
        mock_sandbox = MagicMock()
        mock_sandbox.work_dir = '/tmp/sandbox'
        mock_config.templates.global_rules = '/dev/null'
        pass

    def test_restore_from_state_valid(self, mock_config):
        gcri = MagicMock(spec=GCRI)
        task_dict = {
            'task': 'Test Task',
            'memory': StructuredMemory().model_dump(),
            'feedback': 'Good',
            'count': 1
        }
        memory, feedback, start_index = GCRI._restore_from_state(gcri, task_dict, None)
        assert feedback == 'Good'
        assert start_index == 2
        assert isinstance(memory, StructuredMemory)

    def test_restore_from_state_invalid(self):
        gcri = MagicMock(spec=GCRI)
        invalid_state = {'memory': {'history': 'not-a-list'}}
        with pytest.raises(ValueError):
            GCRI._restore_from_state(gcri, invalid_state, None)

    def test_handle_iteration_success_manual_commit(self, mock_config):
        gcri = MagicMock(spec=GCRI)
        gcri.sandbox = MagicMock()
        gcri.callbacks = MagicMock()
        gcri.sandbox.work_dir = '/tmp/test'
        gcri.sandbox.get_winning_branch_path.return_value = '/tmp/branch/1'
        result = {
            'best_branch_index': 0,
            'final_output': 'Done'
        }
        gcri.callbacks.on_commit_request.return_value = True
        GCRI._handle_iteration_success(gcri, result, 1, 'manual')
        gcri.sandbox.commit_winning_branch.assert_called_once_with('/tmp/branch/1')

    def test_handle_iteration_success_manual_reject(self, mock_config):
        gcri = MagicMock(spec=GCRI)
        gcri.sandbox = MagicMock()
        gcri.callbacks = MagicMock()
        gcri.sandbox.get_winning_branch_path.return_value = '/tmp/branch/1'
        gcri.callbacks.on_commit_request.return_value = False
        GCRI._handle_iteration_success(gcri, {'best_branch_index': 0}, 1, 'manual')
        gcri.sandbox.commit_winning_branch.assert_not_called()
