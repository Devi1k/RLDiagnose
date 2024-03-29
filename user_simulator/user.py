# -*- coding:utf-8 -*-

import copy
import json
import os.path
import random

import dialogue_configuration
from utils.word_match import *


# from data.goal_set import goal_set


class User(object):
    def __init__(self, parameter, model):
        self.max_turn = parameter["max_turn"]
        self.parameter = parameter
        self.allow_wrong_service = parameter.get("allow_wrong_service")
        self.word_dict = load_dict()
        self.model = model
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        with open(os.path.join(data_path, 'similar.json'), 'r') as f:
            self.similarity_dict = json.load(f)

        self._init()

    def _init(self):
        """
        used for initializing an instance or an episode.
        :return: Nothing
        """
        self.state = {
            "turn": 0,
            "action": None,
            "request_slots": {},
            "inform_slots": {},
            "explicit_inform_slots": {},  # For slots that belong to goal["explicit_inform_slots"]
            "implicit_inform_slots": {}  # For slots that belong to goal["implicit_inform_slots"]
        }
        if self.parameter['train_mode'] == 1:
            with open('data/no_noise_data/goal_set_train.json', 'r') as f:
                goal_set = json.load(f)
            self.goal_id = random.randint(0, len(goal_set) - 1)
            self.goal_set = goal_set[str(self.goal_id)]
        else:
            with open('data/no_noise_data/goal_set_test.json', 'r') as f:
                goal_set = json.load(f)
            first_index = int(list(goal_set.keys())[0])
            self.goal_id = random.randint(0, len(goal_set) - 1) + first_index
            self.goal_set = goal_set[str(self.goal_id)]
        # self.goal_id = random.randint(0, len(goal_set) - 1)
        # self.goal_set = goal_set[str(self.goal_id)]

        self.goal = self.goal_set["goal"]
        seg_list = copy.deepcopy(self.goal['explicit_inform_slots'])
        tmp_list = replace_list(seg_list, self.word_dict, model=self.model, similarity_dict=self.similarity_dict)
        tmp_exp = dict()
        for r in tmp_list:
            tmp_exp[r] = True
        self.goal['explicit_inform_slots'] = tmp_exp
        # print(self.goal)
        self.episode_over = False
        self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_NOT_COME_YET

    def get_goal(self):
        return copy.deepcopy(self.goal), self.goal_id

    def initialize(self):
        self._init()
        self.state["action"] = "request"
        self.state["request_slots"]["service"] = dialogue_configuration.VALUE_UNKNOWN
        for slot in self.goal["explicit_inform_slots"].keys():
            self.state["inform_slots"][slot] = self.goal["explicit_inform_slots"][slot]
        user_action = self._assemble_user_action()
        return user_action

    def _assemble_user_action(self):
        user_action = {
            "speaker": "user",
            "action": self.state["action"],
            "request_slots": self.state["request_slots"],
            "inform_slots": self.state["inform_slots"],
            "explicit_inform_slots": self.state["explicit_inform_slots"],
            "implicit_inform_slots": self.state["implicit_inform_slots"],
            "turn": self.state["turn"]
        }
        return user_action

    def next(self, agent_action, turn):
        agent_act_type = agent_action["action"]
        self.state["turn"] = turn
        if self.state["turn"] == self.max_turn - 2:
            self.episode_over = True
            self.state["action"] = dialogue_configuration.CLOSE_DIALOGUE
            self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_FAILED

        if self.episode_over is not True:
            if agent_act_type == "request":
                self._response_request(agent_action=agent_action)
            elif agent_act_type == "inform":
                self._response_inform(agent_action=agent_action)
        user_action = self._assemble_user_action()
        reward = self._reward_function()
        return user_action, reward, self.episode_over, self.dialogue_status

    def _response_request(self, agent_action):
        for slot in agent_action["request_slots"].keys():
            # todo: Whether it should be judged as implicit inform slot
            if slot in self.goal["implicit_inform_slots"].keys():  # inform right max_slot
                # if slot in self.goal["explicit_inform_slots"].keys():  # inform right max_slot
                self.state["action"] = "inform"
                # self.state["inform_slots"][slot] = self.goal["explicit_inform_slots"][slot]
                self.state["inform_slots"][slot] = self.goal["implicit_inform_slots"][slot]
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_INFORM_RIGHT_SLOT
            else:
                self.state["action"] = "inform"
                self.state["inform_slots"][slot] = False
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_SLOT
                implicit_inform = list(self.goal["implicit_inform_slots"].keys())
                N = random.randint(0, 5)
                if N == 0:
                    pass
                else:
                    for i in range(N - 1):
                        slot = random.choice(implicit_inform)
                        self.state["inform_slots"][slot] = self.goal["implicit_inform_slots"][slot]
                        implicit_inform.remove(slot)
            # elif slot not in self.goal[
            #     "max_slot"].keys():  # agent request wrong max_slot,user inform implicit_inform_slots
            # self.state["action"] = "inform"
            # self.state["inform_slots"][slot] = False
            # self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_SLOT
            # # user再说句话或者不说
            # implicit_inform = list(self.goal["implicit_inform_slots"].keys())
            # N = random.randint(0, 5)
            # if N == 0:
            #     pass
            # else:
            #     for i in range(N - 1):
            #         slot = random.choice(implicit_inform)
            #         self.state["inform_slots"][slot] = self.goal["implicit_inform_slots"][slot]
            #         implicit_inform.remove(slot)

    def _response_inform(self, agent_action):
        user_all_inform_slots = copy.deepcopy(self.goal["explicit_inform_slots"])
        user_all_inform_slots.update(self.goal["implicit_inform_slots"])

        # The agent informed the right service and dialogue is over.
        if "service" in agent_action["inform_slots"].keys() and agent_action["inform_slots"]["service"] == self.goal[
            "service_tag"]:
            self.state["action"] = dialogue_configuration.CLOSE_DIALOGUE
            self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_SUCCESS
            self.episode_over = True
            self.state["inform_slots"].clear()
            self.state["request_slots"].pop("service")
        elif "service" in agent_action["inform_slots"].keys() and agent_action["inform_slots"]["service"] != self.goal[
            "service_tag"]:
            # The user denys the informed service, and the dialogue will going on.
            if self.allow_wrong_service == 1:
                # 这里要结合agent的改，agent inform了一个错的服务，要怎么办，我感觉要么失败要么是inform了一个错的max_slot啊
                self.state["action"] = dialogue_configuration.DENY
                # 之后再改吧，反正这种应该不会出现
                self.state["inform_slots"]["service"] = agent_action["inform_slots"]["service"]
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_SERVICE
            else:
                self.state["action"] = dialogue_configuration.CLOSE_DIALOGUE
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_FAILED
                self.episode_over = True
                self.state["inform_slots"].clear()

    def _reward_function(self):
        if self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_NOT_COME_YET:
            return dialogue_configuration.REWARD_FOR_NOT_COME_YET
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
            return dialogue_configuration.REWARD_FOR_DIALOGUE_SUCCESS
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_FAILED:
            return dialogue_configuration.REWARD_FOR_DIALOGUE_FAILED
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_SERVICE:
            return dialogue_configuration.REWARD_FOR_INFORM_WRONG_SERVICE
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_RIGHT_SLOT:
            return dialogue_configuration.REWARD_FOR_INFORM_RIGHT_SLOT
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_SLOT:
            return dialogue_configuration.REWARD_FOR_INFORM_WRONG_SLOT
