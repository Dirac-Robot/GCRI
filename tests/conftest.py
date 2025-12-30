import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


@pytest.fixture
def mock_config(tmp_path):
    from ato.adict import ADict

    template_dir = tmp_path / 'templates'
    template_dir.mkdir()

    templates = {}
    for phase, keyword in [
        ('strategy_generator', 'strategy'),
        ('hypothesis', 'hypothesis'),
        ('reasoning', 'reasoning'),
        ('verification', 'verify'),
        ('decision_maker', 'decision'),
        ('memory_manager', 'memory'),
        ('active_memory', 'active_memory')
    ]:
        tpl = template_dir / f'dummy_{phase}.j2'
        tpl.write_text(f'Mock Template for {phase}: {keyword}')
        templates[phase] = str(tpl)

    global_rules = template_dir / 'global_rules.md'
    global_rules.write_text('Mock Global Rules')

    base_agent_config = {'model_id': 'gpt-4o', 'parameters': {}, 'temperature': 0.7}

    return ADict({
        'project_dir': str(tmp_path / 'project'),
        'run_dir': str(tmp_path / 'run'),
        'protocols': {
            'max_iterations': 3,
            'max_tries_per_agent': 1,
            'aggregate_targets': ['hypothesis', 'reasoning', 'counter_example', 'counter_strength', 'adjustment'],
            'accept_all': False
        },
        'templates': {
            'global_rules': str(global_rules),
            'strategy_generator': templates['strategy_generator'],
            'hypothesis': templates['hypothesis'],
            'reasoning': templates['reasoning'],
            'verification': templates['verification'],
            'decision': templates['decision_maker'],
            'memory': templates['memory_manager'],
            'active_memory': templates['active_memory'],
        },
        'agents': {
            'strategy_generator': base_agent_config,
            'branches': [{
                'hypothesis': base_agent_config,
                'reasoning': base_agent_config,
                'verification': base_agent_config
            }],
            'decision': base_agent_config,
            'memory': base_agent_config,
        }
    })
