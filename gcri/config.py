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
            max_tokens=25600
        ),
        gcri_options=ADict(
            use_web_search=True
        )
    )
    config.agents.compression = dict(
        model_id='gpt-5-mini',
        parameters=ADict(
            max_tokens=25600
        )
    )
    config.agents.strategy_generator = dict(
        model_id='gpt-5-mini',
        parameters=ADict(
            max_tokens=25600
        ),
        gcri_options=ADict(
            use_web_search=True
        )
    )
    config.agents.branches = [
        {
            agent_name: ADict(
                model_id='gpt-5-mini',
                parameters=dict(
                    max_tokens=25600
                ),
                gcri_options=ADict(
                    use_code_tools=True,
                    use_web_search=True,
                    max_recursion_depth=None
                )
            ) for agent_name in AGENT_NAMES_IN_BRANCH
        } for _ in range(3)
    ]
    config.agents.decision = ADict(
        model_id='gpt-5.2',
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
        model_id='gpt-5-mini',
        parameters=ADict(
            max_tokens=25600
        ),
        gcri_options=ADict(
            use_code_tools=True,
            use_web_search=True,
            max_recursion_depth=None
        )
    )
    config.templates = dict(
        planner=get_template_path('planner.txt', 'v0.1.0'),
        compression=get_template_path('compression.txt', 'v0.1.0'),
        black_and_white_lists=get_template_path('black_and_white_lists.json', 'v0.1.0'),
        strategy_generator=get_template_path('strategy_generator.txt', 'v0.1.0'),
        hypothesis=get_template_path('hypothesis.txt', 'v0.1.0'),
        reasoning=get_template_path('reasoning.txt', 'v0.1.0'),
        verification=get_template_path('verification.txt', 'v0.1.0'),
        decision=get_template_path('decision.txt', 'v0.1.0'),
        memory=get_template_path('memory.txt', 'v0.1.0'),
        active_memory=get_template_path('active_memory.txt', 'v0.1.0'),
        global_rules=get_template_path('global_rules.txt', 'v0.1.0')
    )
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


@scope.observe()
def reduce_tokens(config):
    config.templates = dict(
        planner=get_template_path('planner.txt', 'v0.1.1'),
        compression=get_template_path('compression.txt', 'v0.1.1'),
        black_and_white_lists=get_template_path('black_and_white_lists.json', 'v0.1.1'),
        strategy_generator=get_template_path('strategy_generator.txt', 'v0.1.1'),
        hypothesis=get_template_path('hypothesis.txt', 'v0.1.1'),
        reasoning=get_template_path('reasoning.txt', 'v0.1.1'),
        verification=get_template_path('verification.txt', 'v0.1.1'),
        decision=get_template_path('decision.txt', 'v0.1.1'),
        memory=get_template_path('memory.txt', 'v0.1.1'),
        active_memory=get_template_path('active_memory.txt', 'v0.1.1'),
        global_rules=get_template_path('global_rules.txt', 'v0.1.1')
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


@scope.observe(lazy=True)
def no_web_search(config):
    for agent_name, agent_info in config.agents.items():
        if agent_name == 'branches':
            for branch_info in agent_info:
                for branch_agent_name, branch_agent_info in branch_info.items():
                    if 'gcri_options' in branch_agent_info:
                        branch_agent_info.gcri_options.update(
                            use_web_search=False
                        )
        else:
            if 'gcri_options' in agent_info:
                agent_info.gcri_options.update(
                    use_web_search=False
                )


@scope.observe(lazy=True)
def no_code_tools(config):
    for agent_name, agent_info in config.agents.items():
        if agent_name == 'branches':
            for branch_info in agent_info:
                for branch_agent_name, branch_agent_info in branch_info.items():
                    if 'gcri_options' in branch_agent_info:
                        branch_agent_info.gcri_options.update(
                            use_code_tools=False
                        )
        else:
            if 'gcri_options' in agent_info:
                agent_info.gcri_options.update(
                    use_code_tools=False
                )


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
                    max_recursion_depth=None
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
                    max_recursion_depth=None
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
                    max_recursion_depth=None
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
                    max_recursion_depth=None
                )
            ) for agent_name in AGENT_NAMES_IN_BRANCH
        } for _ in range(3)
    ]


@scope.observe()
def local_qwen(config):
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
                        max_recursion_depth=None
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


@scope.observe()
def local_llama(config):
    config.agents.endpoint_url = 'http://localhost:8000/v1'

    with scope.lazy():
        config.agents.planner = dict(
            model_id='neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8',
            parameters=dict(
                max_tokens=25600,
                model_provider='openai',
                base_url=config.agents.endpoint_url,
                api_key='EMPTY',
                temperature=0
            )
        )
        config.agents.compression = dict(
            model_id='neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8',
            parameters=dict(
                max_tokens=25600,
                model_provider='openai',
                base_url=config.agents.endpoint_url,
                api_key='EMPTY',
                temperature=0
            )
        )
        config.agents.strategy_generator = dict(
            model_id='neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8',
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
                    model_id='neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8',
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
                        max_recursion_depth=None
                    )
                ) for agent_name in AGENT_NAMES_IN_BRANCH
            } for _ in range(3)
        ]
        config.agents.decision = dict(
            model_id='neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8',
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
            model_id='neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8',
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
