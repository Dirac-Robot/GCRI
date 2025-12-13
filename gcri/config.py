import os
from importlib import resources
from pathlib import Path

from ato.adict import ADict
from ato.scope import Scope
from loguru import logger

scope = Scope(config=ADict.auto())
AGENT_NAMES_IN_BRANCH = ['hypothesis', 'reasoning', 'verification']


def get_template_path(file_path: str) -> str:
    try:
        with resources.path('gcri.templates', file_path) as path:
            return str(path)
    except (ImportError, TypeError, ModuleNotFoundError):
        current_dir = Path(__file__).resolve().parent
        path = current_dir/'templates'/file_path
        if path.exists():
            return str(path)
        raise FileNotFoundError(f'Template not found: {file_path}')


@scope.observe(default=True)
def default(config):
    config.custom_config_path = None
    config.agents.planner = dict(
        model_id='gpt-5.2',
        parameters=dict(
            max_tokens=25600
        )
    )
    config.agents.compression = dict(
        model_id='gpt-5-mini',
        parameters=dict(
            max_tokens=25600
        )
    )
    config.agents.strategy_generator = dict(
        model_id='gpt-5-mini',
        parameters=dict(
            max_tokens=25600
        )
    )
    config.agents.branches = [
        {
            agent_name: dict(
                model_id='gpt-5-mini',
                parameters=dict(
                    max_tokens=25600
                ),
                gcri_options=dict(
                    use_code_tools=True,
                    use_web_search=True,
                    max_recursion_depth=30
                )
            ) for agent_name in AGENT_NAMES_IN_BRANCH
        } for _ in range(3)
    ]
    config.agents.decision = dict(
        model_id='gpt-5.2',
        parameters=dict(
            max_tokens=25600
        ),
        gcri_options=dict(
            use_code_tools=True,
            use_web_search=True
        )
    )
    config.agents.memory = dict(
        model_id='gpt-5-mini',
        parameters=dict(
            max_tokens=25600
        ),
        gcri_options=dict(
            use_code_tools=True,
            use_web_search=True
        )
    )
    config.templates = dict(
        planner=get_template_path('planner.txt'),
        compression=get_template_path('compression.txt'),
        black_and_white_lists=get_template_path('black_and_white_lists.json'),
        strategy_generator=get_template_path('strategy_generator.txt'),
        hypothesis=get_template_path('hypothesis.txt'),
        reasoning=get_template_path('reasoning.txt'),
        verification=get_template_path('verification.txt'),
        decision=get_template_path('decision.txt'),
        memory=get_template_path('memory.txt'),
        active_memory=get_template_path('active_memory.txt'),
    )
    config.plan.num_max_tasks = 3
    config.max_iterations = 3
    config.protocols = dict(
        accept_all=True,
        aggregate_targets=['strategy', 'hypothesis', 'counter_example', 'adjustment', 'counter_strength'],
        max_tries_per_agent=3
    )
    config.use_deep_feedback = False
    config.log_dir = './gcri_logs'


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
def large_models(config):
    config.agents.branches = [
        dict(
            hypothesis=dict(
                model_id='gpt-5.2',
                parameters=dict(
                    max_tokens=25600
                ),
                gcri_options=dict(
                    use_code_tools=True,
                    use_web_search=True,
                    max_recursion_depth=30
                )
            ),
            reasoning=dict(
                model_id='gpt-5.2',
                parameters=dict(
                    max_tokens=25600
                ),
                gcri_options=dict(
                    use_code_tools=True,
                    use_web_search=True,
                    max_recursion_depth=30
                )
            ),
            verification=dict(
                model_id='gpt-5-mini',
                parameters=dict(
                    max_tokens=25600
                ),
                gcri_options=dict(
                    use_code_tools=True,
                    use_web_search=True,
                    max_recursion_depth=30
                )
            )
        ) for _ in range(3)
    ]


@scope.observe()
def gpt_4_1_based(config):
    config.agents.branches = [
        {
            agent_name: dict(
                model_id='gpt-4.1',
                parameters=dict(
                    max_tokens=25600
                ),
                gcri_options=dict(
                    use_code_tools=True,
                    use_web_search=True,
                    max_recursion_depth=30
                )
            ) for agent_name in AGENT_NAMES_IN_BRANCH
        } for _ in range(3)
    ]


@scope.observe()
def local_llm_based(config):
    config.agents.endpoint_url = 'http://localhost:8000/v1'

    with scope.lazy():
        config.agents.planner = dict(
            model_id='Qwen/Qwen2.5-72B-Instruct',
            parameters=dict(
                max_tokens=25600,
                model_provider='openai',
                base_url=config.agents.endpoint_url,
                api_key='EMPTY',
                temperature=0
            )
        )
        config.agents.compression = dict(
            model_id='Qwen/Qwen2.5-72B-Instruct',
            parameters=dict(
                max_tokens=25600,
                model_provider='openai',
                base_url=config.agents.endpoint_url,
                api_key='EMPTY',
                temperature=0
            )
        )
        config.agents.strategy_generator = dict(
            model_id='Qwen/Qwen2.5-72B-Instruct',
            parameters=dict(
                max_tokens=25600,
                model_provider='openai',
                base_url=config.agents.endpoint_url,
                api_key='EMPTY',
                temperature=0
            )
        )
        config.agents.branches = [
            {
                agent_name: dict(
                    model_id='Qwen/Qwen2.5-72B-Instruct',
                    parameters=dict(
                        max_tokens=25600,
                        model_provider='openai',
                        base_url=config.agents.endpoint_url,
                        api_key='EMPTY',
                        temperature=0
                    ),
                    gcri_options=dict(
                        use_code_tools=True,
                        use_web_search=True,
                        max_recursion_depth=30
                    )
                ) for agent_name in AGENT_NAMES_IN_BRANCH
            } for _ in range(3)
        ]
        config.agents.decision = dict(
            model_id='Qwen/Qwen2.5-72B-Instruct',
            parameters=dict(
                max_tokens=25600,
                model_provider='openai',
                base_url=config.agents.endpoint_url,
                api_key='EMPTY',
                temperature=0
            ),
            gcri_options=dict(
                use_code_tools=True,
                use_web_search=True
            )
        )
        config.agents.memory = dict(
            model_id='Qwen/Qwen2.5-72B-Instruct',
            parameters=dict(
                max_tokens=25600,
                model_provider='openai',
                base_url=config.agents.endpoint_url,
                api_key='EMPTY',
                temperature=0
            ),
            gcri_options=dict(
                use_code_tools=True,
                use_web_search=True
            )
        )
