# -*- coding:utf-8 -*-
"""
An RL-based agent which learns policy using actor-critic algorithm.
"""

import os
import sys

sys.path.append(os.getcwd().replace("src/dialogue_system/agent", ""))

from policy_learning.actor_critic import ActorCritic
from agent.agent import Agent


class AgentActorCritic(Agent):
    def __init__(self, parameter):
        super(AgentActorCritic, self).__init__(parameter=parameter)
        input_size = parameter["input_size"]
        output_size = len(self.action_space)
        self.actor_critic = ActorCritic(n_features=input_size, n_actions=output_size, param=parameter)

    def next(self, state, turn, greedy_strategy=None, episode_over=False):
        self.agent_action["turn"] = turn
        state_rep = self.state_to_representation_last(state=state)  # sequence representation.
        action_index = self.actor_critic.actor.select_action(state_rep)
        agent_action = self.action_space[action_index]

        agent_action["turn"] = turn
        agent_action["speaker"] = "agent"

        return agent_action, action_index

    def train(self, trajectories):
        self.actor_critic.train(trajectories=trajectories)

    def update_target_network(self):
        pass

    def save_model(self, model_performance, episodes_index, checkpoint_path=None):
        self.actor_critic.save_model(model_performance=model_performance, episodes_index=episodes_index,
                            checkpoint_path=checkpoint_path)
