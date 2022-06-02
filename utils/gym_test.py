import time
from collections import deque

import gym
import torch

from policy_learning.actor_critic import ActorCritic
from utils.config import get_config
from utils.zfilter import ZFilter


def simulation_epoch(env, agent, running_state, parameter, device):
    log = dict()
    total_reward = 0
    i_episode = 0
    MAX_EPISODE = parameter.get("MAX_EPISODE")
    RENDER = parameter.get("RENDER")
    trajectory_pool = deque(maxlen=48)

    for i_episode in range(MAX_EPISODE):
        # s = env.reset()
        # t = 0
        # track_r = []
        s = env.reset()
        trajectory = []
        if running_state is not None:
            s = running_state(s)
        reward_episode = 0
        while True:
            a = int(agent.actor.select_action(s).detach().cpu().numpy())
            s_, r, done, info = env.step(a)
            reward_episode += r
            if done:
                r = -20
            if running_state is not None:
                s_ = running_state(s_)

            mask = 0 if done else 1

            trajectory.append([s, a, mask, s_, r])
            # memory.push([s, a, mask, s_, r])
            # track_r.append(r)
            if RENDER:
                env.render()
            if done:
                trajectory_pool.append(trajectory)
                break
            s = s_

        total_reward += reward_episode
    log['num_episodes'] = i_episode
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / i_episode
    # trajectories
    trajectory_pool = list(trajectory_pool)
    batch_size = 30
    # batch = memory.sample()
    for index in range(0, len(trajectory_pool), batch_size):
        stop = max(len(trajectory_pool), index + batch_size)
        batch_trajectory = trajectory_pool[index:stop]
        agent.train(trajectories=batch_trajectory)
    return log


def eval(env, agent, running_state, parameter, device):
    log = dict()
    total_reward = 0
    i_episode = 0
    MAX_EPISODE = parameter.get("MAX_EPISODE")
    RENDER = parameter.get("RENDER")
    trajectory_pool = deque(maxlen=48)

    for i_episode in range(MAX_EPISODE):
        # s = env.reset()
        # t = 0
        # track_r = []
        s = env.reset()
        trajectory = []
        if running_state is not None:
            s = running_state(s)
        reward_episode = 0
        while True:
            # s = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            a = int(agent.actor.select_action(s).detach().cpu().numpy())
            s_, r, done, info = env.step(a)
            reward_episode += r
            if done:
                r = -20
            if running_state is not None:
                s_ = running_state(s_)

            mask = 0 if done else 1

            trajectory.append([s, a, mask, s_, r])
            # track_r.append(r)
            if RENDER:
                env.render()
            if done:
                trajectory_pool.append(trajectory)
                break
            s = s_

        total_reward += reward_episode
    log['num_episodes'] = i_episode
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / i_episode
    return log


def run():
    env = gym.make('CartPole-v0')
    env.seed(1)  # reproducible
    env = env.unwrapped

    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n

    running_state = ZFilter((N_F,), clip=5)
    config_file = '../settings_ac.yaml'
    parameter = get_config(config_file)
    iter_nums = parameter.get("ITER_NUMS")
    log_interval = parameter.get("LOG_INTERVAL")
    device = torch.device('cuda',
                          index=parameter.get("gpu_index", 1)) if torch.cuda.is_available() else torch.device(
        'cpu')
    agent = ActorCritic(n_features=N_F, n_actions=N_A, param=parameter)
    for i in range(iter_nums):
        t0 = time.time()
        log = simulation_epoch(env, agent, running_state, parameter, device)
        t1 = time.time()
        log_eval = eval(env, agent, running_state, parameter, device)
        t2 = time.time()
        if i % log_interval == 0:
            print('{}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                i, t1 - t0, t2 - t1, log['avg_reward'], log_eval['avg_reward']))


if __name__ == '__main__':
    run()
