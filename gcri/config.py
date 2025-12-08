from ato.adict import ADict
from ato.scope import Scope

scope = Scope(config=ADict.auto())


@scope.observe(default=True)
def default(config):
    config.agents.strategy_generator = dict(
        model_id='gpt-5-mini',
        parameters=dict()
    )
    config.agents.branches = [
        dict(
            hypothesis=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            ),
            reasoning=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            ),
            verification=dict(
                model_id='gpt-5-mini',
                parameters=dict(),
                options=dict(
                    use_code_tools=True
                )
            )
        ),
        dict(
            hypothesis=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            ),
            reasoning=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            ),
            verification=dict(
                model_id='gpt-5-mini',
                parameters=dict(),
                options=dict(
                    use_code_tools=True
                )
            )
        ),
        dict(
            hypothesis=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            ),
            reasoning=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            ),
            verification=dict(
                model_id='gpt-5-mini',
                parameters=dict(),
                options=dict(
                    use_code_tools=True
                )
            )
        )
    ]
    config.agents.decision = dict(
        model_id='gpt-5.1',
        parameters=dict(),
        options=dict(
            use_code_tools=True
        )
    )
    config.agents.compression = dict(
        model_id='gpt-5-nano',
        parameters=dict()
    )
    config.templates = dict(
        strategy_generator='./gcri/templates/strategy_generator.txt',
        hypothesis='./gcri/templates/hypothesis.txt',
        reasoning='./gcri/templates/reasoning.txt',
        verification='./gcri/templates/verification.txt',
        decision='./gcri/templates/decision.txt',
        compression='./gcri/templates/compression.txt',
        compression_prev='./gcri/templates/compression_prev.txt'
    )
    config.max_iterations = 5
    config.protocols = dict(
        accept_all=True,
        aggregate_targets=['strategy', 'hypothesis', 'adjustment', 'counter_strength'],
        max_tries_per_agent=3
    )
    config.use_deep_feedback = False
    config.log_dir = './gcri_logs'


@scope.observe()
def faster(config):
    config.agents.branches = [
        dict(
            hypothesis=dict(
                model_id='gpt-5-mini',
                parameters=dict(
                    reasoning=dict(effort='low')
                )
            ),
            reasoning=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            ),
            verification=dict(
                model_id='gpt-5-mini',
                parameters=dict(
                    reasoning=dict(effort='low')
                )
            )
        ),
        dict(
            hypothesis=dict(
                model_id='gpt-5-mini',
                parameters=dict(
                    reasoning=dict(effort='low')
                )
            ),
            reasoning=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            ),
            verification=dict(
                model_id='gpt-5-mini',
                parameters=dict(
                    reasoning=dict(effort='low')
                )
            )
        ),
        dict(
            hypothesis=dict(
                model_id='gpt-5-mini',
                parameters=dict(
                    reasoning=dict(effort='low')
                )
            ),
            reasoning=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            ),
            verification=dict(
                model_id='gpt-5-mini',
                parameters=dict(
                    reasoning=dict(effort='low')
                )
            )
        )
    ]


@scope.observe()
def mix_up(config):
    config.agents.decision = dict(
        model_id='gemini-2.5-flash-lite',
        parameters=dict(temperature=0.5)
    )
    config.agents.branches = [
        dict(
            hypothesis=dict(
                model_id='gemini-2.5-flash-lite',
                parameters=dict()
            ),
            reasoning=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            ),
            verification=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            )
        ),
        dict(
            hypothesis=dict(
                model_id='gemini-2.5-flash-lite',
                parameters=dict()
            ),
            reasoning=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            ),
            verification=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            )
        ),
        dict(
            hypothesis=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            ),
            reasoning=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            ),
            verification=dict(
                model_id='gpt-5-mini',
                parameters=dict()
            )
        )
    ]