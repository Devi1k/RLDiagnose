# coding:utf-8
# 生成goal_set  我用的傻逼方法🥲：生成goal_set.txt，然后复制到goal_set.py
# 不过目前训练dqn的话可以直接用goal_set.py，之后新增了数据需要生成goal_set的时候再说
# （因为这种方法没办法很聪明地生成"护照"那几个业务的goal_set）
import copy
import random

from configuration import service, slot_max, requirement_weight

requirementall = []
for i in range(len(requirement_weight)):
    requirementall.append([])
for i in range(len(requirement_weight)):
    requirementall[i] = list(requirement_weight[i].keys())
f = open("goal_set.txt", 'w')


def pick_slot(requirement):
    requirement.remove(requirement[0])
    slot = random.sample(requirement, k=8)
    return slot[0], slot[1], slot[2], slot[3], slot[4], slot[5], slot[6], slot[7]


def generate_goalset():
    for i in range(400, 2000):
        consult_id = i
        n = random.randint(0, len(service) - 1)
        service_tag = service[n]
        max_slot = slot_max[n]
        requirement = copy.deepcopy(requirementall[n])
        slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8 = pick_slot(requirement)
        f.write(f'{{\'consult_id\':\'{consult_id}\',\n')
        f.write("'goal':{'request_slots':{'service':'UNK'},\n")
        f.write(f'\'max_slot\':{{\'{max_slot}\':True}},\n')
        f.write(f'\'service_tag\':\'{service_tag}\',\n')
        f.write(f'\'explicit_inform_slots\':{{\'{slot1}\':True,\'{slot2}\':True,\'{slot3}\':True,\'{slot4}\':True}},\n')
        f.write(
            f'\'implicit_inform_slots\':{{\'{slot5}\':True,\'{slot6}\':True,\'{slot7}\':True,\'{slot8}\':True}}}}}},\n')
    f.close()


if __name__ == "__main__":
    generate_goalset()
