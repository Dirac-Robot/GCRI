import json
import os
from datetime import datetime
from functools import partial

from langchain.chat_models import init_chat_model
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send
from loguru import logger

from gcri.graphs.schemas import Verification, Reasoning, Hypothesis, Decision, Strategies, Compression
from gcri.graphs.states import TaskState, BranchState, HypothesisResult


class GCRIGraph:
    def __init__(self, config):
        self.config = config
        graph = StateGraph(TaskState)
        branch = StateGraph(BranchState)
        branch.add_node('sample_hypothesis', self.sample_hypothesis)
        branch.add_node('reasoning_and_refine', self.reasoning_and_refine)
        branch.add_node('verify', self.verify)
        branch.add_edge(START, 'sample_hypothesis')
        branch.add_edge('sample_hypothesis', 'reasoning_and_refine')
        branch.add_edge('reasoning_and_refine', 'verify')
        branch.add_edge('verify', END)
        branch_workflow = branch.compile()
        graph.add_node('branch_executor', branch_workflow)
        strategy_generator_config = config.agents.strategy_generator
        strategy_agent = init_chat_model(
            strategy_generator_config.model_id,
            **strategy_generator_config.parameters
        )
        decision_config = config.agents.decision
        decision_agent = init_chat_model(
            decision_config.model_id,
            **decision_config.parameters
        )
        compression_config = config.agents.compression
        compression_agent = init_chat_model(
            compression_config.model_id,
            **compression_config.parameters
        )
        graph.add_node('sample_strategies', partial(self.sample_strategies, agent=strategy_agent))
        graph.add_node('refine', partial(self.sample_strategies, agent=strategy_agent))
        graph.add_node('aggregate', self.aggregate)
        graph.add_node('decision', partial(self.decide, agent=decision_agent))
        graph.add_node('compression', partial(self.compress, agent=compression_agent))
        graph.add_edge(START, 'sample_strategies')
        graph.add_conditional_edges(
            'sample_strategies',
            self.map_branches,
            ['branch_executor']
        )
        graph.add_edge('branch_executor', 'aggregate')
        graph.add_edge('aggregate', 'decision')
        graph.add_edge('decision', 'compression')
        graph.add_edge('compression', END)
        self._graph = graph
        self._workflow = graph.compile()
        log_dir = os.path.join(config.log_dir, datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.log_dir = log_dir

    @property
    def graph(self):
        return self._graph

    @property
    def workflow(self):
        return self._workflow

    def map_branches(self, state: TaskState):
        num_branches = min(len(self.config.agents.branches), len(state.strategies))
        return [
            Send(
                'branch_executor',
                {
                    'index': index,
                    'original_task': state.task,
                    'strategy': state.strategies[index],
                    'feedback': state.feedback
                }
            ) for index in range(num_branches)
        ]

    def aggregate(self, state: TaskState):
        aggregated_results = [
            {
                key: getattr(result, key, '')
                for key in self.config.protocols.aggregate_targets
            }
            for result in state.results
            if result.counter_strength != 'strong' or self.config.protocols.accept_all
        ]
        return {'aggregated_result': json.dumps(aggregated_results, indent=4, ensure_ascii=False)}

    def sample_strategies(self, state: TaskState, agent):
        logger.info('Request generating strategies...')
        template_path = self.config.templates.strategy_generator
        with open(template_path, 'r') as f:
            template = f.read().format(
                task=state.task,
                compressed_output=state.compressed_output,
                feedback=state.feedback,
                num_hypothesis=len(self.config.agents.branches)
            )
        for _ in range(self.config.protocols.max_tries_per_agent):
            strategies = agent.with_structured_output(schema=Strategies).invoke(template)
            if strategies is not None:
                break
        else:
            raise ValueError(
                f'Agent could not generate strategies '
                f'for {self.config.protocols.max_tries_per_agent} times.'
            )
        for index, strategy in enumerate(strategies.strategies):
            logger.info(f'Sampled strategy #{index}: {strategy}')
        return dict(strategies=strategies.strategies)

    def sample_hypothesis(self, state: BranchState):
        logger.info(f'Request sampling hypothesis for strategy #{state.index}...')
        hypothesis_config = self.config.agents.branches[state.index].hypothesis
        agent = init_chat_model(hypothesis_config.model_id, **hypothesis_config.parameters)
        template_path = self.config.templates.hypothesis
        with open(template_path, 'r') as f:
            template = f.read().format(task=state.original_task, strategy=state.strategy)
        for _ in range(self.config.protocols.max_tries_per_agent):
            hypothesis = agent.with_structured_output(schema=Hypothesis).invoke(template)
            if hypothesis is not None:
                break
        else:
            raise ValueError(
                f'Agent could not generate hypothesis '
                f'for {self.config.protocols.max_tries_per_agent} times '
                f'at strategy #{state.index}.'
            )
        logger.info(f'Sampled hypothesis #{state.index}: {hypothesis.hypothesis}')
        return dict(hypothesis=hypothesis.hypothesis)

    def reasoning_and_refine(self, state: BranchState):
        logger.info(f'Request reasoning and refining hypothesis #{state.index}...')
        reasoning_config = self.config.agents.branches[state.index].reasoning
        agent = init_chat_model(reasoning_config.model_id, **reasoning_config.parameters)
        template_path = self.config.templates.reasoning
        with open(template_path, 'r') as f:
            template = f.read().format(
                task=state.original_task,
                strategy=state.strategy,
                hypothesis=state.hypothesis
            )
        for _ in range(self.config.protocols.max_tries_per_agent):
            reasoning = agent.with_structured_output(schema=Reasoning).invoke(template)
            if reasoning is not None:
                break
        else:
            raise ValueError(
                f'Agent could not generate refined hypothesis '
                f'for {self.config.protocols.max_tries_per_agent} times '
                f'at hypothesis #{state.index}.'
            )
        logger.info(f'Refined hypothesis #{state.index}: {reasoning.refined_hypothesis}')
        return dict(
            reasoning=reasoning.reasoning,
            hypothesis=reasoning.refined_hypothesis
        )

    def improve(self, state: TaskState, agent):
        pass

    def verify(self, state: BranchState):
        logger.info(f'Request verifying refined hypothesis #{state.index}...')
        verification_config = self.config.agents.branches[state.index].verification
        agent = init_chat_model(verification_config.model_id, **verification_config.parameters)
        template_path = self.config.templates.verification
        with open(template_path, 'r') as f:
            template = f.read()
        template = template.format(
            task=state.original_task,
            strategy=state.strategy,
            reasoning=state.reasoning,
            hypothesis=state.hypothesis
        )
        for _ in range(self.config.protocols.max_tries_per_agent):
            verification = agent.with_structured_output(schema=Verification).invoke(template)
            if verification is not None:
                break
        else:
            raise ValueError(
                f'Agent could not generate verification '
                f'for {self.config.protocols.max_tries_per_agent} times.'
            )
        result = HypothesisResult(
            index=state.index,
            strategy=state.strategy,
            reasoning=state.reasoning,
            hypothesis=state.hypothesis,
            counter_reasoning=verification.reasoning,
            counter_example=verification.counter_example,
            counter_strength=verification.counter_strength,
            adjustment=verification.adjustment
        )
        logger.info(f'Counter-Example of Hypothesis #{state.index}: {verification.counter_example}')
        return {'results': [result]}

    def decide(self, state: TaskState, agent):
        logger.info(f'Request generating final decision for current loop...')
        template_path = self.config.templates.decision
        with open(template_path, 'r') as f:
            template = f.read()
        template = template.format(task=state.task, aggregated_result=state.aggregated_result)
        for _ in range(self.config.protocols.max_tries_per_agent):
            decision = agent.with_structured_output(schema=Decision).invoke(template)
            if decision is not None:
                break
        else:
            raise ValueError(
                f'Agent could not generate decision '
                f'for {self.config.protocols.max_tries_per_agent} times.'
            )
        logger.info(f'Decision: {decision.decision}')
        if not decision.decision:
            logger.info(f'Feedback: {decision.feedback}')
        return {
            'decision': decision.decision,
            'final_output': decision.final_output if decision.decision else decision.feedback,
            'feedback': decision.feedback,
            'decision_reasoning': decision.decision_reasoning
        }

    def compress(self, state: TaskState, agent):
        logger.info(f'Request compression...')
        if state.decision:
            template_path = self.config.templates.compression
        else:
            template_path = self.config.templates.compression_prev
        with open(template_path, 'r') as f:
            template = f.read()
        template = template.format(
            task=state.task,
            final_output=state.final_output,
            aggregated_result=state.aggregated_result,
            feedback=state.feedback
        )
        for _ in range(self.config.protocols.max_tries_per_agent):
            compression = agent.with_structured_output(schema=Compression).invoke(template)
            if compression is not None:
                break
        else:
            raise ValueError(
                f'Agent could not generate compression '
                f'for {self.config.protocols.max_tries_per_agent} times.'
            )
        logger.info(f'Compressed Result: {compression.compressed_output}')
        return {'compressed_output': compression.compressed_output}

    def __call__(self, task):
        feedback = ''
        compressed_output = ''
        result = None
        for index in range(self.config.max_iterations):
            result = self.workflow.invoke({'task': task, 'feedback': feedback, 'compressed_output': compressed_output})
            result['results'] = [dict(state) for state in result['results']]
            os.makedirs(self.log_dir, exist_ok=True)
            log_path = os.path.join(self.log_dir, f'log_iteration_{index:02d}.json')
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            logger.info(f'Result of iteration {index} saved to: {log_path}')
            if result['decision']:
                logger.info('Final result is successfully deduced.')
                return result
            else:
                compressed_output = result['compressed_output']
                feedback = result['feedback']
        logger.info('Final result is not deduced, but iteration count is over.')
        return result
