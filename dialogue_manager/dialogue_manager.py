# -*- coding:utf-8 -*-

import random
from collections import deque

import dialogue_configuration
from agent.agent_dqn import AgentDQN
from state_tracker.state_tracker import StateTracker


class DialogueManager(object):
    """
    Dialogue manager of this dialogue system.
    """

    def __init__(self, user, agent, parameter):
        self.state_tracker = StateTracker(user=user, agent=agent, parameter=parameter)
        self.parameter = parameter
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
        self.inform_wrong_service_count = 0

    def initialize(self, train_mode=1, epoch_index=None):
        self.state_tracker.initialize()
        self.inform_wrong_service_count = 0
        user_action = self.state_tracker.user.initialize()
        self.state_tracker.state_updater(user_action=user_action)
        self.state_tracker.agent.initialize()
        state = self.state_tracker.get_state()
        agent_action, action_index = self.state_tracker.agent.next(state=state, turn=self.state_tracker.turn,
                                                                   greedy_strategy=greedy_strategy)
        self.state_tracker.state_updater(agent_action=agent_action)
        # state = self.state_tracker.get_state()
        # self.self.log.info(state["current_slots"]["agent_request_slots"].keys())  #测试是否为空，证明不是
        return agent_action

    def set_agent(self, agent):
        self.state_tracker.set_agent(agent=agent)

    def next(self, save_record, train_mode, agent_action, greedy_strategy):
        """
        The next two turns of this dialogue session. The agent will take action first and then followed by user simulator.
        :param save_record: bool, save record?
        :param train_mode: int, 1: the purpose of simulation is to train the model, 0: just for simulation and the
                                   parameters of the model will not be updated.
        :return: immediate reward for taking this agent action.
        """
        # state = self.state_tracker.get_state()
        # # Agent takes action.
        # agent_action, action_index = self.state_tracker.agent.next(state=state, turn=self.state_tracker.turn,
        #                                                            greedy_strategy=greedy_strategy)
        # self.state_tracker.state_updater(agent_action=agent_action)
        # User takes action.
        user_action, reward, episode_over, dialogue_status = self.state_tracker.user.next(agent_action=agent_action,
                                                                                          turn=self.state_tracker.turn)
        self.state_tracker.state_updater(user_action=user_action)
        state = self.state_tracker.get_state()
        agent_action, action_index = self.state_tracker.agent.next(state=state, turn=self.state_tracker.turn,
                                                                   greedy_strategy=greedy_strategy)
        if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_SERVICE:
            self.inform_wrong_service_count += 1
        if save_record is True:
            self.record_training_sample(
                state=state,
                agent_action=action_index,
                next_state=self.state_tracker.get_state(),
                reward=reward,
                episode_over=episode_over
            )
        else:
            pass

        self.state_tracker.state_updater(agent_action=agent_action)

        return reward, episode_over, dialogue_status, agent_action

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over):
        state = self.state_tracker.agent.state_to_representation_last(state)
        next_state = self.state_tracker.agent.state_to_representation_last(next_state)
        self.experience_replay_pool.append((state, agent_action, reward, next_state, episode_over))  # 每两个turn添加一次

    def train(self):  # 一个epoch一次
        if isinstance(self.state_tracker.agent, AgentDQN):
            self.__train_dqn()
            self.state_tracker.agent.update_target_network()

    def __train_dqn(self):  # 一个epoch中 len(self.experience_replay_pool) / (batch_size) 次
        """
        Train dqn.
        :return:
        """
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size", 16)
        for iter in range(int(len(self.experience_replay_pool) / (batch_size))):
            batch = random.sample(self.experience_replay_pool, batch_size)  # sample()用于随机抽样。
            loss = self.state_tracker.agent.train(batch=batch)
            cur_bellman_err += loss["loss"]
        print("cur bellman err %.4f, experience replay pool %s" % (
            float(cur_bellman_err) / len(self.experience_replay_pool), len(self.experience_replay_pool)))
        # print("cur bellman err %.4f)
