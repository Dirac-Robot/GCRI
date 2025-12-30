import pytest
from unittest.mock import MagicMock, patch
from gcri.graphs.gcri_unit import GCRI
from gcri.graphs.states import TaskState, StructuredMemory, Strategy
from tests.mocks.agents import ScenarioBasedMockLLM
from tests.mocks.sandbox import MockSandboxManager


class TestGCRILoop:
    @pytest.fixture
    def mock_sandbox(self):
        return MockSandboxManager()

    @pytest.fixture
    def mock_llm_happy_path(self):
        scenarios = [
            {
                'trigger': 'strategy',
                'response': {
                    'strategies': [{
                        'name': 'bfs',
                        'description': 'desc',
                        'feedback_reflection': 'None',
                        'hints': ['hint1']
                    }],
                    'strictness': 'moderate',
                    'intent_analysis': 'intent'
                }
            },
            {
                'trigger': 'hypothesis',
                'response': {
                    'hypothesis': "print('Hello World')"
                }
            },
            {
                'trigger': 'reasoning',
                'response': {
                    'reasoning': 'Good reasoning',
                    'refined_hypothesis': "print('Hello World')"
                }
            },
            {
                'trigger': 'verify',
                'response': {
                    'counter_example': 'None',
                    'counter_strength': 'none',
                    'adjustment': 'None',
                    'reasoning': 'Valid'
                }
            },
            {
                'trigger': 'decision',
                'response': {
                    'decision': True,
                    'best_branch_index': 0,
                    'global_feedback': 'Success',
                    'branch_evaluations': [{
                        'branch_index': 0,
                        'summary_hypothesis': 'Print Hello',
                        'summary_counter_example': 'None',
                        'status': 'valid',
                        'failure_category': 'none',
                        'reasoning': 'Works'
                    }],
                    'final_output': 'Code works'
                }
            }
        ]
        return ScenarioBasedMockLLM(scenarios, default_response='{}')

    @patch('gcri.graphs.gcri_unit.SandboxManager')
    @patch('gcri.graphs.gcri_unit.build_decision_model')
    @patch('gcri.graphs.gcri_unit.build_model')
    def test_gcri_loop_happy_path(self, mock_build_model, mock_build_decision, MockSandboxClass, mock_config, mock_llm_happy_path):
        mock_sandbox = MockSandboxManager()
        MockSandboxClass.return_value = mock_sandbox

        from tests.mocks.agents import StructuredMockWrapper

        def create_mock_agent(*args, **kwargs):
            agent_mock = MagicMock()
            def with_structured_output(schema=None):
                return StructuredMockWrapper(mock_llm_happy_path, schema)
            agent_mock.with_structured_output = with_structured_output
            return agent_mock

        mock_build_model.side_effect = create_mock_agent
        mock_build_decision.side_effect = create_mock_agent

        gcri = GCRI(mock_config)
        result = gcri(
            task='Write a python script that prints Hello World',
            commit_mode='auto-accept'
        )

        assert result is not None, 'GCRI should return a result dict'
        assert result['decision'] is True
        assert result['final_output'] == 'Code works'
        assert result['best_branch_index'] == 0

        call_count = len(mock_llm_happy_path._call_log)
        assert call_count >= 5, f'Expected at least 5 LLM calls, got {call_count}'
        assert len(mock_sandbox.setup_branch_calls) >= 1, 'setup_branch should be called for each branch'

        commit_calls = [c for c in mock_sandbox.command_history if 'COMMIT' in str(c)]
        assert len(commit_calls) == 1, f'Expected 1 commit, got {len(commit_calls)}'

    def test_gcri_loop_failure_path(self):
        pass
