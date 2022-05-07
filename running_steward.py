# -*-coding: utf-8 -*-

import copy
import json
import pickle
from collections import deque

import dialogue_configuration
from agent.agent_rule import AgentRule
from dialogue_manager.dialogue_manager import DialogueManager
from user_simulator.user import User


class RunningSteward(object):
    """
    The steward of running the dialogue system.
    """

    def __init__(self, parameter, checkpoint_path):
        self.epoch_size = parameter.get("epoch_size")
        self.parameter = parameter
        user = User(parameter=parameter)
        agent = AgentRule(parameter=parameter)
        self.dialogue_manager = DialogueManager(user=user, agent=agent, parameter=parameter)
        self.best_result = {"success_rate": 0.0, "average_reward": 0.0, "average_turn": 0, "average_wrong_service": 10}
        self.checkpoint_path = checkpoint_path
        self.learning_curve = {}

    def simulation_epoch(self, epoch_size, train_mode):
        """
        Simulating one epoch when training model.
        :param epoch_size: the size of each epoch, i.e., the number of dialogue sessions of each epoch.
        :return: a dict of simulation results including success rate, average reward, average number of wrong services.
        """
        success_count = 0
        absolute_success_count = 0
        total_reward = 0
        total_turns = 0
        inform_wrong_service_count = 0
        for epoch_index in range(0, epoch_size, 1):
            agent_action, action_index, prev_state = self.dialogue_manager.initialize(
                train_mode=self.parameter.get("train_mode"))
            episode_over = False
            while episode_over is False:
                # reward, episode_over, dialogue_status = self.dialogue_manager.next(save_record=True,
                #                                                                    train_mode=train_mode,
                #                                                                    greedy_strategy=1)
                reward, episode_over, dialogue_status, _agent_action, _action_index, _prev_state = self.dialogue_manager.next(
                    save_record=True,
                    train_mode=train_mode,
                    greedy_strategy=1,
                    prev_agent_action=agent_action,
                    prev_agent_index=action_index,
                    prev_state=prev_state)
                agent_action = _agent_action
                action_index = _action_index
                prev_state = _prev_state
                total_reward += reward
            total_turns += self.dialogue_manager.state_tracker.turn
            inform_wrong_service_count += self.dialogue_manager.inform_wrong_service_count
            if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                success_count += 1
                if self.dialogue_manager.inform_wrong_service_count == 0:
                    absolute_success_count += 1
            goal = self.dialogue_manager.state_tracker.user.get_goal()
            item = json.dumps(goal, ensure_ascii=False, indent=4)
            index = json.dumps(epoch_index)
            if epoch_index == 0:
                if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                    with open("success.json", "w", encoding='utf-8') as f:
                        f.write("goal_set=[\n")
                        f.write("{'consult_id':" + index + ",\n")
                        f.write("'goal':" + item + "},\n")
                if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_FAILED:
                    with open("fail.json", "w", encoding='utf-8') as f:
                        f.write("goal_set=[\n")
                        f.write("{'consult_id':" + index + ",\n")
                        f.write("'goal':" + item + ",\n")
            elif epoch_index == epoch_size - 1:
                if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                    with open("success.json", "w+", encoding='utf-8') as f:
                        f.write("{'consult_id':" + index + ",\n")
                        f.write("'goal':" + item + "}]\n")
                if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_FAILED:
                    with open("fail.json", "w+", encoding='utf-8') as f:
                        f.write("{'consult_id':" + index + ",\n")
                        f.write("'goal':" + item + "]\n")
            else:
                if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                    with open("success.json", "w+", encoding='utf-8') as f:
                        f.write("{'consult_id':" + index + ",\n")
                        f.write("'goal':" + item + "},\n")
                if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_FAILED:
                    with open("fail.json", "w+", encoding='utf-8') as f:
                        f.write("{'consult_id':" + index + ",\n")
                        f.write("'goal':" + item + ",\n")

        success_rate = float("%.3f" % (float(success_count) / epoch_size))
        absolute_success_rate = float("%.3f" % (float(absolute_success_count) / epoch_size))
        average_reward = float("%.3f" % (float(total_reward) / epoch_size))
        average_turn = float("%.3f" % (float(total_turns) / epoch_size))
        average_wrong_service = float("%.3f" % (float(inform_wrong_service_count) / epoch_size))
        res = {"success_rate": success_rate, "average_reward": average_reward, "average_turn": average_turn,
               "average_wrong_service": average_wrong_service, "ab_success_rate": absolute_success_rate}
        return res

    def warm_start(self, agent, epoch_number):
        """
        Warm-starting the dialogue, using the sample from rule-based agent to fill the experience replay pool for DQN.
        :param agent: the agent used to warm start dialogue system.
        :param epoch_number: the number of epoch when warm starting, and the number of dialogue sessions of each epoch
        equals to the simulation epoch.   #lyj: the number of dialogue sessions of each epoch equals to the simulation epoch.就是epoch size?
        :return: nothing to return.
        """
        self.dialogue_manager.set_agent(agent=agent)
        for index in range(0, epoch_number, 1):
            res = self.simulation_epoch(epoch_size=self.epoch_size, train_mode=1)
            print("%3d simulation SR %s, ABSR %s,ave reward %s, ave turns %s, ave wrong service %s" % (
                index, res['success_rate'], res["ab_success_rate"], res['average_reward'], res['average_turn'],
                res["average_wrong_service"]))

    def simulate(self, agent, epoch_number, train_mode):
        """
        Simulating between agent and user simulator.
        :param agent: the agent used to simulate, an instance of class Agent.
        :param epoch_number: the epoch number of simulation.
        :param train_mode: int, 1: the purpose of simulation is to train the model, 0: just for simulation and the
                           parameters of the model will not be updated.
        :return: nothing to return.
        """
        # save_model = self.parameter.get("save_model")  #要在参数那里设路径
        self.dialogue_manager.set_agent(agent=agent)
        if train_mode == 1:
            for index in range(0, epoch_number, 1):
                # Training AgentDQN with experience replay
                # if train_mode == 1 and isinstance(self.dialogue_manager.state_tracker.agent, AgentDQN):
                # Simulating and filling experience replay pool.
                # res = self.simulation_epoch(epoch_size=self.epoch_size, train_mode=train_mode)
                self.dialogue_manager.train()
                res = self.simulation_epoch(epoch_size=self.epoch_size, train_mode=train_mode)
                print("Train %3d simulation SR %s, ABSR %s,ave reward %s, ave turns %s, ave wrong service %s" % (
                    index, res['success_rate'], res["ab_success_rate"], res['average_reward'], res['average_turn'],
                    res["average_wrong_service"]))
                result = self.evaluate_model(index)
                if result["success_rate"] >= self.best_result["success_rate"] and \
                        result["average_wrong_service"] <= self.best_result[
                    "average_wrong_service"] and train_mode == 1:
                    self.dialogue_manager.experience_replay_pool = deque(
                        maxlen=self.parameter.get("experience_replay_pool_size"))
                    self.simulation_epoch(epoch_size=self.epoch_size, train_mode=train_mode)
                    self.dialogue_manager.state_tracker.agent.save_model(model_performance=result, episodes_index=index,
                                                                         checkpoint_path=self.checkpoint_path)
                    print("The model was saved.")
                    self.best_result = copy.deepcopy(result)
        else:
            self.simulation_epoch(epoch_size=self.epoch_size, train_mode=train_mode)

    def evaluate_model(self, index):
        """
        Evaluating model during training.
        :param index: int, the simulation index.
        :return: a dict of evaluation results including success rate, average reward, average number of wrong services.
        """
        save_performance = self.parameter.get("save_performance")

        train_mode = self.parameter.get("train_mode")
        success_count = 0
        absolute_success_count = 0
        total_reward = 0
        total_truns = 0
        evaluate_epoch_number = self.parameter.get("evaluate_epoch_number")
        # evaluate_epoch_number = len(self.dialogue_manager.state_tracker.user.goal_set["test"])
        inform_wrong_service_count = 0
        for epoch_index in range(0, evaluate_epoch_number, 1):
            agent_action, action_index, prev_state = self.dialogue_manager.initialize(
                train_mode=self.parameter.get("train_mode"))
            episode_over = False
            while episode_over is False:
                reward, episode_over, dialogue_status, _agent_action, _action_index, _prev_state = self.dialogue_manager.next(
                    save_record=True,
                    train_mode=train_mode,
                    greedy_strategy=1,
                    prev_agent_action=agent_action,
                    prev_agent_index=action_index,
                    prev_state=prev_state)
                agent_action = _agent_action
                action_index = _action_index
                prev_state = _prev_state
                total_reward += reward

            total_truns += self.dialogue_manager.state_tracker.turn
            inform_wrong_service_count += self.dialogue_manager.inform_wrong_service_count
            if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                success_count += 1
                if self.dialogue_manager.inform_wrong_service_count == 0:
                    absolute_success_count += 1
        success_rate = float("%.3f" % (float(success_count) / evaluate_epoch_number))
        absolute_success_rate = float("%.3f" % (float(absolute_success_count) / evaluate_epoch_number))
        average_reward = float("%.3f" % (float(total_reward) / evaluate_epoch_number))
        average_turn = float("%.3f" % (float(total_truns) / evaluate_epoch_number))
        average_wrong_service = float("%.3f" % (float(inform_wrong_service_count) / evaluate_epoch_number))
        res = {"success_rate": success_rate, "average_reward": average_reward, "average_turn": average_turn,
               "average_wrong_service": average_wrong_service, "ab_success_rate": absolute_success_rate}
        self.learning_curve.setdefault(index, dict())
        self.learning_curve[index]["success_rate"] = success_rate
        self.learning_curve[index]["average_reward"] = average_reward
        self.learning_curve[index]["average_turn"] = average_turn
        self.learning_curve[index]["average_wrong_service"] = average_wrong_service
        if index % 10 == 0:
            self.__print_run_info__()
        if index % 10 == 9 and save_performance == 1:
            self.__dump_performance__(epoch_index=index)
        print("Eval %3d simulation SR %s, ABSR %s, ave reward %s, ave turns %s, ave wrong service %s" % (
            index, res['success_rate'], res["ab_success_rate"], res['average_reward'], res['average_turn'],
            res["average_wrong_service"]))
        return res

    def __dump_performance__(self, epoch_index):

        lr = self.parameter.get("dqn_learning_rate")
        # reward_for_success = self.parameter.get("reward_for_success")
        # reward_for_fail = self.parameter.get("reward_for_fail")
        # reward_for_not_come_yet = self.parameter.get("reward_for_not_come_yet")
        # reward_for_inform_right_symptom = self.parameter.get("reward_for_inform_right_symptom")
        reward_for_success = dialogue_configuration.REWARD_FOR_DIALOGUE_SUCCESS
        reward_for_fail = dialogue_configuration.REWARD_FOR_DIALOGUE_FAILED
        reward_for_not_come_yet = dialogue_configuration.REWARD_FOR_NOT_COME_YET
        reward_for_inform_right_symptom = dialogue_configuration.REWARD_FOR_INFORM_RIGHT_SLOT

        max_turn = self.parameter.get("max_turn")
        # minus_left_slots = self.parameter.get("minus_left_slots")
        gamma = self.parameter.get("gamma")
        epsilon = self.parameter.get("epsilon")
        run_id = self.parameter.get('run_id')

        file_name = "learning_rate_d" + "_e" + "_agent" + "_T" + str(max_turn) + "_lr" + str(lr) + "_RFS" + str(
            reward_for_success) + \
                    "_RFF" + str(reward_for_fail) + "_RFNCY" + str(reward_for_not_come_yet) + "_RFIRS" + str(
            reward_for_inform_right_symptom) + "_gamma" + str(gamma) + "_epsilon" + str(
            epsilon) + "_RID" + str(run_id) + "_" + str(epoch_index) + ".p"

        pickle.dump(file=open(self.parameter.get("performance_save_path") + file_name, "wb"), obj=self.learning_curve)

    def __print_run_info__(self):
        # print(json.dumps(self.parameter, indent=2))
        # agent_id = self.parameter.get("agent_id")
        # dqn_id = self.parameter.get("dqn_id")
        # service_number = self.parameter.get("service_number")
        lr = self.parameter.get("dqn_learning_rate")
        # reward_for_success = self.parameter.get("reward_for_success")
        # reward_for_fail = self.parameter.get("reward_for_fail")
        # reward_for_not_come_yet = self.parameter.get("reward_for_not_come_yet")
        # reward_for_inform_right_symptom = self.parameter.get("reward_for_inform_right_symptom")
        reward_for_success = dialogue_configuration.REWARD_FOR_DIALOGUE_SUCCESS
        reward_for_fail = dialogue_configuration.REWARD_FOR_DIALOGUE_FAILED
        reward_for_not_come_yet = dialogue_configuration.REWARD_FOR_NOT_COME_YET
        reward_for_inform_right_symptom = dialogue_configuration.REWARD_FOR_INFORM_RIGHT_SLOT

        max_turn = self.parameter.get("max_turn")
        # minus_left_slots = self.parameter.get("minus_left_slots")
        gamma = self.parameter.get("gamma")
        epsilon = self.parameter.get("epsilon")
        # data_set_name = self.parameter.get("goal_set").split("/")[-2]
        # run_id = self.parameter.get('run_id')
        info = "learning_rate_d" + "_T" + str(max_turn) + "_lr" + str(lr) + "_RFS" + str(reward_for_success) + \
               "_RFF" + str(reward_for_fail) + "_RFNCY" + str(reward_for_not_come_yet) + "_RFIRS" + str(
            reward_for_inform_right_symptom) + "_gamma" + str(gamma) + "_epsilon" + str(epsilon)

        print("[INFO]:", info)
