# -*- coding: utf-8 -*-

import copy
import json

import dialogue_configuration
from agent.agent import Agent
from data.configuration import requirement_weight, service


class AgentRule(Agent):
    def __init__(self, parameter):
        super(AgentRule, self).__init__(parameter=parameter)
        with open('data/slot_max_weight.json', 'r') as f:
            slot_max_weight = json.load(f)
        self.slot_max = list(slot_max_weight.keys())

    def next(self, state, turn, greedy_strategy, episode_over=False):
        score_max = 0
        max = 0
        score = []
        for i in range(len(self.slot_max)):
            score.append(0)
        inform_slots = list(state["current_slots"]["inform_slots"].keys())
        for i in range(len(self.slot_max)):
            for j in range(len(inform_slots)):
                if inform_slots[j] in requirement_weight[i].keys():  # 如果该事项包含的slots中有inform的slot，该事项的score就加上该slot的权重
                    if state["current_slots"]["inform_slots"][inform_slots[j]] is True:
                        score[i] += requirement_weight[i][inform_slots[j]]
                    elif state["current_slots"]["inform_slots"][inform_slots[j]] is False:
                        score[i] -= requirement_weight[i][inform_slots[j]]
                else:
                    pass
                # todo: 如果有一样的怎么办
            if score[i] > score_max:  # 然后比较score的值，选出最大的那个，用max记下score最大的位置
                score_max = score[i]
                max = i
        candidate_service = service[max]
        candidate_requirement = self.slot_max[max]
        self.agent_action["request_slots"].clear()
        self.agent_action["inform_slots"].clear()
        self.agent_action["turn"] = turn

        if episode_over:
            self.agent_action['action'] = dialogue_configuration.CLOSE_DIALOGUE
            self.agent_action['inform_slots']: {}
            self.agent_action['request_slots']: {}
            agent_action = copy.deepcopy(self.agent_action)
            agent_action.pop("turn")
            agent_index = self.action_space.index(agent_action)
            return self.agent_action, agent_index
        if score_max > 100:
            self.agent_action["action"] = "inform"
            self.agent_action["inform_slots"]["service"] = candidate_service
        else:
            requirement = candidate_requirement
            self.agent_action["action"] = "request"
            self.agent_action["request_slots"][requirement] = dialogue_configuration.VALUE_UNKNOWN
        agent_action = copy.deepcopy(self.agent_action)
        agent_action.pop("turn")
        agent_index = self.action_space.index(agent_action)
        return self.agent_action, agent_index  # 区分这里的self.agent_action和agent_action
