# -*- coding:utf-8 -*-
import copy
import random
from collections import deque

import dialogue_configuration
from agent.agent_actor_critic import AgentActorCritic
from agent.agent_dqn import AgentDQN
from policy_learning.PrioritizedReplay import PrioritizedReplayBuffer
from state_tracker.state_tracker import StateTracker


class DialogueManager(object):
    """
    Dialogue manager of this dialogue system.
    """

    def __init__(self, user, agent, parameter):
        self.state_tracker = StateTracker(user=user, agent=agent, parameter=parameter)
        self.parameter = parameter
        # self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))

        self.inform_wrong_service_count = 0
        if self.parameter['agent_id'] == 2:
            self.trajectory = []
            self.trajectory_pool = deque(maxlen=self.parameter["trajectory_pool_size"])
        elif self.parameter['agent_id'] == 1:
            if self.parameter['prioritized_replay']:
                self.experience_replay_pool = PrioritizedReplayBuffer(
                    buffer_size=self.parameter["experience_replay_pool_size"])
            else:
                self.experience_replay_pool = deque(maxlen=self.parameter["experience_replay_pool_size"])

    def initialize(self, train_mode=1, epoch_index=None, greedy_strategy=1):
        self.state_tracker.initialize()
        self.inform_wrong_service_count = 0
        user_action = self.state_tracker.user.initialize()
        self.state_tracker.state_updater(user_action=user_action)
        self.state_tracker.agent.initialize()
        init_state = self.state_tracker.get_state()
        agent_action, action_index = self.state_tracker.agent.next(state=init_state, turn=self.state_tracker.turn,
                                                                   greedy_strategy=greedy_strategy)
        self.state_tracker.state_updater(agent_action=agent_action)
        # state = self.state_tracker.get_state()
        # self.self.log.info(state["current_slots"]["agent_request_slots"].keys())  #测试是否为空，证明不是
        return agent_action, action_index, init_state

    def set_agent(self, agent):
        self.state_tracker.set_agent(agent=agent)

    def next(self, prev_state, save_record, train_mode, prev_agent_action, prev_agent_index, greedy_strategy):

        user_action, reward, episode_over, dialogue_status = self.state_tracker.user.next(
            agent_action=prev_agent_action,
            turn=self.state_tracker.turn)
        self.state_tracker.state_updater(user_action=user_action)
        _state = self.state_tracker.get_state()
        if user_action['action'] == 'closing':
            if save_record is True:
                if self.parameter['agent_id'] == 1 and self.parameter['prioritized_replay']:
                    current_action_value = self.state_tracker.agent.current_action_value
                    target_action_value = self.state_tracker.agent.next_state_values_DDQN(prev_state)
                    TD_error = reward + self.parameter["gamma"] * target_action_value - current_action_value
                    self.record_prioritized_training_sample(
                        state=prev_state,
                        agent_action=prev_agent_index,
                        next_state=_state,
                        reward=reward,
                        episode_over=episode_over,
                        TD_error=TD_error
                    )
                else:
                    self.record_training_sample(
                        state=prev_state,
                        agent_action=prev_agent_index,
                        next_state=_state,
                        reward=reward,
                        episode_over=episode_over
                    )
            else:
                pass

            if episode_over is True and self.parameter['agent_id'] == 2:
                self.trajectory_pool.append(copy.deepcopy(self.trajectory))
            return reward, episode_over, dialogue_status, prev_agent_action, prev_agent_index, _state
        # Agent takes action.
        agent_action, action_index = self.state_tracker.agent.next(state=_state, turn=self.state_tracker.turn,
                                                                   greedy_strategy=greedy_strategy,
                                                                   episode_over=episode_over)
        self.state_tracker.state_updater(agent_action=agent_action)
        if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_SERVICE:
            self.inform_wrong_service_count += 1

        if save_record is True:
            if self.parameter['agent_id'] == 1 and self.parameter['prioritized_replay']:
                current_action_value = self.state_tracker.agent.current_action_value
                target_action_value = self.state_tracker.agent.next_state_values_DDQN(prev_state)
                TD_error = reward + self.parameter["gamma"] * target_action_value - current_action_value
                self.record_prioritized_training_sample(
                    state=prev_state,
                    agent_action=prev_agent_index,
                    next_state=_state,
                    reward=reward,
                    episode_over=episode_over,
                    TD_error=TD_error
                )
            else:
                self.record_training_sample(
                    state=prev_state,
                    agent_action=prev_agent_index,
                    next_state=_state,
                    reward=reward,
                    episode_over=episode_over
                )
        else:
            pass

        if episode_over is True and self.parameter['agent_id'] == 2:
            self.trajectory_pool.append(copy.deepcopy(self.trajectory))
        prev_state = _state

        return reward, episode_over, dialogue_status, agent_action, action_index, prev_state

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over):
        state = self.state_tracker.agent.state_to_representation_last(state)
        next_state = self.state_tracker.agent.state_to_representation_last(next_state)
        if self.parameter['agent_id'] == 1:
            self.experience_replay_pool.append((state, agent_action, reward, next_state, episode_over))
        elif self.parameter['agent_id'] == 2:  # 每两个turn添加一次
            self.trajectory.append((state, agent_action, reward, next_state, episode_over))

    def record_prioritized_training_sample(self, state, agent_action, reward, next_state, episode_over, TD_error):
        state_rep = self.state_tracker.agent.state_to_representation_last(state=state)
        next_state_rep = self.state_tracker.agent.state_to_representation_last(state=next_state)

        self.experience_replay_pool.add(state_rep, agent_action, reward, next_state_rep, episode_over, TD_error)
        self.state_tracker.agent.action_visitation_count.setdefault(agent_action, 0)
        self.state_tracker.agent.action_visitation_count[agent_action] += 1
        # self.experience_replay_pool.add(
        #     self.state_tracker.agent.record_prioritized_training_sample(state, agent_action, reward, next_state,
        #                                                                 episode_over, TD_error))

    def train(self):  # 一个epoch一次
        if isinstance(self.state_tracker.agent, AgentDQN):
            self.__train_dqn()
            self.state_tracker.agent.update_target_network()
        elif isinstance(self.state_tracker.agent, AgentActorCritic):
            self.__train_actor_critic()

    def __train_dqn(self):  # 一个epoch中 len(self.experience_replay_pool) / (batch_size) 次
        """
        Train dqn.
        :return:
        """
        cur_bellman_err = 0.0
        batch_size = self.parameter["batch_size"]
        for iter in range(int(len(self.experience_replay_pool) / (batch_size))):
            if self.parameter['prioritized_replay']:
                batch = self.experience_replay_pool.sample(batch_size)
            else:
                batch = random.sample(self.experience_replay_pool, batch_size)  # sample()用于随机抽样。
            loss = self.state_tracker.agent.train(batch=batch)
            cur_bellman_err += loss["loss"]
        print("cur bellman err %.4f, experience replay pool %s" % (
            float(cur_bellman_err) / len(self.experience_replay_pool), len(self.experience_replay_pool)))
        # print("cur bellman err %.4f)

    def __train_actor_critic(self):
        """
        Train actor-critic.
        :return:
        """
        trajectory_pool = list(self.trajectory_pool)
        batch_size = self.parameter["batch_size"]
        for index in range(0, len(self.trajectory_pool), batch_size):
            stop = max(len(self.trajectory_pool), index + batch_size)
            batch_trajectory = trajectory_pool[index:stop]
            self.state_tracker.agent.train(trajectories=batch_trajectory)
