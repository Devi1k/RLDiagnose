# -*- coding:utf-8 -*-

# Max turn.
MAX_TURN = 4

# DIALOGUE STATUS
DIALOGUE_STATUS_FAILED = 0
DIALOGUE_STATUS_SUCCESS = 1
DIALOGUE_STATUS_NOT_COME_YET = -1
DIALOGUE_STATUS_INFORM_WRONG_SERVICE = 2
DIALOGUE_STATUS_INFORM_RIGHT_SLOT = 3
DIALOGUE_STATUS_INFORM_WRONG_SLOT = -2

# 这是医疗代码设的
# REWORD FOR DIFFERENT DIALOGUE STATUS.
# REWARD_FOR_DIALOGUE_FAILED = - MAX_TURN
# REWARD_FOR_DIALOGUE_SUCCESS = 2*MAX_TURN
# REWARD_FOR_NOT_COME_YET = -1
# REWARD_FOR_INFORM_WRONG_SERVICE= -MAX_TURN
# REWARD_FOR_INFORM_RIGHT_SLOT= 0.5*MAX_TURN
# REWARD_FOR_INFORM_WRONG_SLOT=-0.5*MAX_TURN

# 这是老师改的，根据老师的意思，这些值还要改改
REWARD_FOR_DIALOGUE_FAILED = -15
REWARD_FOR_DIALOGUE_SUCCESS = 35
REWARD_FOR_NOT_COME_YET = -10
REWARD_FOR_INFORM_WRONG_SERVICE = -10
REWARD_FOR_INFORM_RIGHT_SLOT = 20
REWARD_FOR_INFORM_WRONG_SLOT = -10

# Special Actions.
CLOSE_DIALOGUE = "closing"
THANKS = "thanks"

# Slot value for unknown.
VALUE_UNKNOWN = "UNK"
