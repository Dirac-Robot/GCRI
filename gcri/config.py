import os
from importlib import resources
from pathlib import Path

from ato.adict import ADict
from ato.scope import Scope
from loguru import logger

scope = Scope(config=ADict.auto())
AGENT_NAMES_IN_BRANCH = ['hypothesis', 'reasoning', 'verification']


def get_template_path(file_path: str, template_version: str) -> str:
    try:
        pkg_path = resources.files('gcri.templates').joinpath(template_version, file_path)
        if pkg_path.is_file():
            return str(pkg_path)
    except (ImportError, TypeError, ModuleNotFoundError, AttributeError):
        pass
    current_dir = Path(__file__).resolve().parent
    path = current_dir/'templates'/template_version/file_path
    if path.exists():
        return str(path)
    raise FileNotFoundError(f'Template not found: {template_version}/{file_path}')


@scope.observe(default=True)
def default(config):
    config.custom_config_path = None
    config.agents.planner = dict(
        model_id='gpt-5.2',
        parameters=ADict(
            max_tokens=25600,
            reasoning_effort='low'
        ),
        gcri_options=ADict(
            use_web_search=True
        )
    )
    config.agents.compression = dict(
        model_id='gpt-5-mini',
        parameters=ADict(
            max_tokens=25600,
            reasoning_effort='low'
        )
    )
    config.agents.strategy_generator = dict(
        model_id='gpt-5-mini',
        parameters=ADict(
            max_tokens=25600,
            reasoning_effort='low'
        ),
        gcri_options=ADict(
            use_web_search=True
        )
    )
    config.agents.decision = ADict(
        model_id='gpt-5.2',
        parameters=ADict(
            max_tokens=25600,
            reasoning_effort='low'
        ),
        gcri_options=ADict(
            use_code_tools=True,
            use_web_search=True,
            max_recursion_depth=None
        )
    )
    config.agents.memory = dict(
        model_id='gpt-5-mini',
        parameters=ADict(
            max_tokens=25600,
            reasoning_effort='low'
        ),
        gcri_options=ADict(
            use_code_tools=True,
            use_web_search=True,
            max_recursion_depth=None
        )
    )
    config.agents.aggregator = dict(
        model_id='gpt-5-mini',
        parameters=ADict(
            max_tokens=25600,
            reasoning_effort='low'
        )
    )
    config.template_version = 'v0.1.1'
    config.plan.num_max_tasks = 5
    config.protocols = dict(
        accept_all=True,
        aggregate_targets=['strategy', 'hypothesis', 'counter_example', 'adjustment', 'counter_strength'],
        max_iterations=5,
        max_tries_per_agent=3,
        max_copy_size=10,
        force_output=False
    )
    config.project_dir = os.path.abspath(os.getcwd())
    config.run_dir = os.path.join(config.project_dir, '.gcri')
    config.dashboard = dict(
        enabled=True,
        host='127.0.0.1',
        port=8000,
        monitor_directories=[]  # User can override this with paths to watch
    )
    config.sandbox = dict(
        image='python:3.11-slim',
        timeout=60,
        memory_limit='512m',
        cpu_limit=1.0,
        network_mode='none'
    )
    config.external_memory = dict(
        enabled=True,
        path=None  # Will be set lazily based on run_dir
    )
    config.num_branches = 2
    config.branches_generator_type = 'default'  # 'default', 'deep', 'shallow'
    config.aggregation = dict(
        max_output_branches=3,
        allow_single_source_passthrough=True
    )
    with scope.lazy():
        config.agents.branches = [
            {
                agent_name: ADict(
                    model_id='gpt-5-mini',
                    parameters=dict(
                        max_tokens=25600,
                        reasoning_effort='low'
                    ),
                    gcri_options=ADict(
                        use_code_tools=True,
                        use_web_search=True,
                        max_recursion_depth=None
                    )
                ) for agent_name in AGENT_NAMES_IN_BRANCH
            } for _ in range(config.num_branches)
        ]
        config.templates = dict(
            planner=get_template_path('planner.txt', config.template_version),
            compression=get_template_path('compression.txt', config.template_version),
            black_and_white_lists=get_template_path('black_and_white_lists.json', config.template_version),
            strategy_generator=get_template_path('strategy_generator.txt', config.template_version),
            hypothesis=get_template_path('hypothesis.txt', config.template_version),
            hypothesis_minimal=get_template_path('hypothesis_minimal.txt', config.template_version),
            reasoning=get_template_path('reasoning.txt', config.template_version),
            verification=get_template_path('verification.txt', config.template_version),
            decision=get_template_path('decision.txt', config.template_version),
            memory=get_template_path('memory.txt', config.template_version),
            active_memory=get_template_path('active_memory.txt', config.template_version),
            sandbox_curator=get_template_path('sandbox_curator.txt', config.template_version),
            global_rules=get_template_path('global_rules.txt', config.template_version),
            aggregator=get_template_path('aggregator.txt', config.template_version),
            external_memory_update=get_template_path('external_memory_update.txt', config.template_version)
        )



@scope.observe(default=True, lazy=True)
def apply_custom_config(config):
    if config.custom_config_path is not None:
        if os.path.exists(config.custom_config_path):
            logger.info(f'Override with custom config: {config.custom_config_path}')
            config.update(ADict.from_file(config.custom_config_path), recurrent=True)
        else:
            logger.warning(f'Cannot find custom config: {config.custom_config_path}')
            logger.warning(f'Fallback to default config...')


@scope.observe()
def no_reasoning(config):
    config.agents.planner = dict(
        model_id='gpt-4.1',
        parameters=ADict(
            max_tokens=25600
        ),
        gcri_options=ADict(
            use_code_tools=True,
            use_web_search=True,
            max_recursion_depth=None
        )
    )
    config.agents.compression = dict(
        model_id='gpt-4.1-mini',
        parameters=ADict(
            max_tokens=25600
        ),
        gcri_options=ADict(
            use_code_tools=True,
            use_web_search=True,
            max_recursion_depth=None
        )
    )
    config.agents.strategy_generator = dict(
        model_id='gpt-4.1-mini',
        parameters=ADict(
            max_tokens=25600
        ),
        gcri_options=ADict(
            use_code_tools=True,
            use_web_search=True,
            max_recursion_depth=None
        )
    )
    config.agents.aggregator = dict(
        model_id='gpt-4.1',
        parameters=ADict(
            max_tokens=25600
        ),
        gcri_options=ADict(
            use_code_tools=True,
            use_web_search=True,
            max_recursion_depth=None
        )
    )
    config.agents.decision = ADict(
        model_id='gpt-4.1',
        parameters=ADict(
            max_tokens=25600
        ),
        gcri_options=ADict(
            use_code_tools=True,
            use_web_search=True,
            max_recursion_depth=None
        )
    )
    config.agents.memory = dict(
        model_id='gpt-4.1-mini',
        parameters=ADict(
            max_tokens=25600
        ),
        gcri_options=ADict(
            use_code_tools=True,
            use_web_search=True,
            max_recursion_depth=None
        )
    )

    with scope.lazy():
        config.agents.branches = [
            {
                agent_name: ADict(
                    model_id='gpt-4.1-mini',
                    parameters=dict(
                        max_tokens=25600
                    ),
                    gcri_options=ADict(
                        use_code_tools=True,
                        use_web_search=True,
                        max_recursion_depth=None
                    )
                ) for agent_name in AGENT_NAMES_IN_BRANCH
            } for _ in range(config.num_branches)
        ]
