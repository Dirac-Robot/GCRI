from ato.adict import ADict
from ato.scope import Scope

scope = Scope(config=ADict.auto())
AGENT_NAMES_IN_BRANCH = ['hypothesis', 'reasoning', 'verification']


@scope.observe(default=True)
def default(config):
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
            use_code_tools=True
        )
    )
    config.agents.memory = dict(
        model_id='gpt-5-mini',
        parameters=dict(
            max_tokens=25600
        ),
        gcri_options=dict(
            use_code_tools=True
        )
    )
    config.templates = dict(
        planner='./gcri/templates/planner.txt',
        compression='./gcri/templates/compression.txt',
        black_and_white_lists='./gcri/templates/black_and_white_lists.json',
        strategy_generator='./gcri/templates/strategy_generator.txt',
        hypothesis='./gcri/templates/hypothesis.txt',
        reasoning='./gcri/templates/reasoning.txt',
        verification='./gcri/templates/verification.txt',
        decision='./gcri/templates/decision.txt',
        memory='./gcri/templates/memory.txt',
        active_memory='./gcri/templates/active_memory.txt'
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
                    max_recursion_depth=30
                )
            ) for agent_name in AGENT_NAMES_IN_BRANCH
        } for _ in range(3)
    ]
