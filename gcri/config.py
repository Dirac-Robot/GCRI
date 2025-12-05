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
    config.agents.decision = dict(
        model_id='gpt-5.1',
        parameters=dict()
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
        compression='./gcri/templates/compression.txt'
    )
    config.max_iterations = 5
    config.reject_if_strong_counter_example_exists = False
    config.log_dir = './gcri_logs'


@scope.observe(default=False)
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
