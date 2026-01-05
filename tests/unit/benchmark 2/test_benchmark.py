import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from gcri.benchmark.server import app, BenchmarkRequest, BenchmarkResponse


class TestBenchmarkServer:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json() == {'status': 'ok'}

    def test_benchmark_request_model(self):
        request = BenchmarkRequest(
            task_id='test_001',
            prompt='What is 2 + 2?',
            metadata={'source': 'unit_test'}
        )
        assert request.task_id == 'test_001'
        assert request.prompt == 'What is 2 + 2?'
        assert request.metadata == {'source': 'unit_test'}

    def test_benchmark_response_model(self):
        response = BenchmarkResponse(
            task_id='test_001',
            answer='4',
            reasoning='Simple arithmetic',
            success=True
        )
        assert response.task_id == 'test_001'
        assert response.answer == '4'
        assert response.success is True

    @patch('gcri.benchmark.server.get_gcri_instance')
    def test_solve_endpoint_success(self, mock_get_gcri, client):
        mock_gcri = MagicMock()
        mock_gcri.return_value = {
            'final_output': '4',
            'feedback': 'Calculated correctly'
        }
        mock_get_gcri.return_value = mock_gcri
        response = client.post(
            '/benchmark/solve',
            json={
                'task_id': 'test_001',
                'prompt': 'What is 2 + 2?'
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data['task_id'] == 'test_001'
        assert data['success'] is True


class TestBenchmarkSolver:
    def test_solver_import(self):
        from gcri.benchmark.solver import gcri_solver
        solver = gcri_solver(endpoint='http://localhost:8001')
        assert solver is not None


class TestBenchmarkTasks:
    def test_task_import(self):
        from gcri.benchmark.tasks import gcri_benchmark, gcri_qa_benchmark, gcri_exact_benchmark
        assert gcri_benchmark is not None
        assert gcri_qa_benchmark is not None
        assert gcri_exact_benchmark is not None

    def test_get_scorer(self):
        from gcri.benchmark.tasks import get_scorer
        exact_scorer = get_scorer('exact')
        includes_scorer = get_scorer('includes')
        assert exact_scorer is not None
        assert includes_scorer is not None
