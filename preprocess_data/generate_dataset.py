# coding:utf-8
# 生成goal_set  我用的傻逼方法🥲：生成goal_set.txt，然后复制到goal_set.py
# 不过目前训练dqn的话可以直接用goal_set.py，之后新增了数据需要生成goal_set的时候再说
# （因为这种方法没办法很聪明地生成"护照"那几个业务的goal_set）
import copy
import json
import random

from data.configuration import service, requirement_weight


# f = open("goal_set.txt", 'w')


def pick_slot(requirement, k=8):
    requirement.remove(requirement[0])
    requirement.remove(requirement[0])
    slot = random.sample(requirement, k=k)
    return slot


def generate_goalset_and_dict(slot_max):
    noise = False
    requirement_all = []
    for i in range(len(requirement_weight)):
        requirement_all.append([])
    for i in range(len(requirement_weight)):
        requirement_all[i] = list(requirement_weight[i].keys())
    dict_list = []

    def make_dict(requirement):
        requirement.remove(requirement[0])
        requirement.remove(requirement[0])
        for _r in requirement:
            dict_list.append(_r + '\n')

    for i in range(len(requirement_all)):
        make_dict(requirement_all[i])

    with open('../data/new_dict.txt', 'w') as f:
        f.writelines(dict_list)
    l = dict()
    k = 0
    for i in range(400, 2000):
        res = dict()
        goal = dict()
        max_slot = dict()
        explicit_inform_slots = dict()
        implicit_inform_slots = dict()

        service_num = random.randint(0, len(service) - 1)
        n_candidate = []
        for j in range(4):
            x = random.randint(0, len(service) - 1)
            while service_num == x:
                x = random.randint(0, len(service) - 1)
            n_candidate.append(x)
        requirement = copy.deepcopy(requirement_all[service_num])
        other_requirement = []
        for j in range(4):
            other_requirement.append(copy.deepcopy(requirement_all[n_candidate[j]]))
        max_slot[slot_max[service_num]] = True

        implicit_slots = pick_slot(requirement, k=4)
        explicit_slot = []
        if noise is False:
            # 直接抽取
            explicit_slot = pick_slot(requirement, k=4)
        else:
            # 加噪声抽取
            for j in range(4):
                for item in pick_slot(other_requirement[j], 4):
                    explicit_slot.append(item)
            explicit_slot = random.sample(explicit_slot, k=4) + pick_slot(requirement, 4)

        # slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8 = pick_slot(requirement)
        goal['request_slots'] = {'service': 'UNK'}
        goal['max_slot'] = max_slot
        goal['service_tag'] = service[service_num]
        for _x, _y in zip(explicit_slot, implicit_slots):
            explicit_inform_slots[_x] = True
            implicit_inform_slots[_y] = True

        goal['explicit_inform_slots'] = explicit_inform_slots
        goal['implicit_inform_slots'] = implicit_inform_slots

        res['consult_id'] = i
        res['goal'] = goal
        l[k] = res
        k += 1
    train_size = round(0.8 * len(l))
    # test_size = len(l) - train_size
    train_data = dict()
    test_data = dict()
    for q in range(train_size):
        train_data[q] = l[q]
    for q in range(train_size, len(l)):
        test_data[q] = l[q]
    if noise is False:
        with open('../data/no_noise_data/goal_set_train.json', 'w') as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open('../data/no_noise_data/goal_set_test.json', 'w') as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    else:
        with open('../data/noise_data/goal_set_train.json', 'w') as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open('../data/noise_data/goal_set_test.json', 'w') as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)


def generate_slot_set():
    dic = {}

    for i in range(len(requirement_weight)):
        dic = {**dic, **requirement_weight[i]}
    new_dict = list(dic.keys())
    slot_set = {}
    keys = new_dict
    v = 0
    for i in keys:
        slot_set[i] = v
        v += 1
    slot_set['service'] = v

    with open('../data/slot_set.json', 'w') as f:
        json.dump(slot_set, f, indent=4, ensure_ascii=False)


def generate_slot_max_weight():
    slot_max_weight = dict()
    slot_max = []
    for i in range(len(requirement_weight)):
        name = list(requirement_weight[i].keys())[1]
        slot_max_weight[name] = 100
        slot_max.append(name)
    with open('../data/slot_max_weight.json', 'w') as f:
        json.dump(slot_max_weight, f, indent=4, ensure_ascii=False)
    return slot_max


if __name__ == "__main__":
    generate_slot_set()
    slot_max = generate_slot_max_weight()
    generate_goalset_and_dict(slot_max)
