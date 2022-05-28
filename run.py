# -*- coding:utf-8 -*-

import argparse
import json
import os.path

from agent.agent_dqn import *
from agent.agent_rule import *
from running_steward import RunningSteward

parser = argparse.ArgumentParser()

parser.add_argument("--simulate_epoch_number", dest="simulate_epoch_number", type=int, default=500,
                    help="the number of simulate epoch.")
parser.add_argument("--epoch_size", dest="epoch_size", type=int, default=900, help="the size of each simulate epoch.")
parser.add_argument("--evaluate_epoch_number", dest="evaluate_epoch_number", type=int, default=50,
                    help="the size of each simulate epoch when evaluation.")
parser.add_argument("--experience_replay_pool_size", dest="experience_replay_pool_size", type=int, default=20000,
                    help="the size of experience replay.")
parser.add_argument("--hidden_size_dqn", dest="hidden_size_dqn", type=int, default=300, help="the hidden_size of DQN.")
parser.add_argument("--warm_start", dest="warm_start", type=int, default=0,
                    help="use rule policy to fill the experience replay buffer at the beginning, 1:True; 0:False")
parser.add_argument("--warm_start_epoch_number", dest="warm_start_epoch_number", type=int, default=20,
                    help="the number of epoch of warm starting.")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=30, help="the batch size when training.")
parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.1, help="the greedy of DQN")
parser.add_argument("--gamma", dest="gamma", type=float, default=1.0, help="The discount factor of immediate reward.")
parser.add_argument("--train_mode", dest="train_mode", type=int, default=1, help="training mode? True:1 or False:0")
parser.add_argument("--prioritized_replay", dest="prioritized_replay", type=bool, default=True,
                    help="whether to use prioritized replay in memory")

# TODO: Save model, performance and dialogue content ? And what is the path if yes? #这部分还没写
parser.add_argument("--save_performance", dest="save_performance", type=int, default=1,
                    help="save the performance? 1:Yes, 0:No")
parser.add_argument("--performance_save_path", dest="performance_save_path", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/dqn/learning_rate/"),
                    help="the folder where learning rate save to, ending with /.")
parser.add_argument("--allow_wrong_service", dest="allow_wrong_service", type=int, default=1,
                    help="Allow the agent to inform wrong service? 1:Yes, 0:No")
parser.add_argument("--dqn_learning_rate", dest="dqn_learning_rate", type=float, default=0.001,
                    help="the learning rate of dqn.")
parser.add_argument("--saved_model", dest="saved_model", type=str,
                    default="/home/yanking/disk1/nizepu/govChatbot/model/dqn/checkpoint/model_d_agent_dqn_s1.0_r36.4_t3.36_wd0.0_e49.pkl")
parser.add_argument("--max_turn", dest="max_turn", type=int, default=8, help="the max turn in one episode.")
parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=1221, help="the input_size of DQN.")
args = parser.parse_args()
parameter = vars(args)
print(json.dumps(parameter, indent=2))

checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/dqn/checkpoint/")
# performance_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/dqn/learning_rate/")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


# if not os.path.exists(performance_save_path):
#     os.makedirs(performance_save_path)

def run():
    steward = RunningSteward(parameter=parameter, checkpoint_path=checkpoint_path)

    warm_start = parameter.get("warm_start")
    warm_start_epoch_number = parameter.get("warm_start_epoch_number")
    train_mode = parameter.get("train_mode")
    simulate_epoch_number = parameter.get("simulate_epoch_number")

    # Warm start.
    if warm_start == 1 and train_mode == 1:
        print("warm starting...")
        agent = AgentRule(parameter=parameter)
        steward.warm_start(agent=agent, epoch_number=warm_start_epoch_number)
    # simulate
    agent = AgentDQN(parameter=parameter)
    steward.simulate(agent=agent, epoch_number=simulate_epoch_number, train_mode=train_mode)


if __name__ == "__main__":
    run()
