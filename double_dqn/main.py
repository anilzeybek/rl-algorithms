import argparse
import random
from time import time

import gym
import numpy as np
import torch

from double_dqn_agent import DoubleDQNAgent


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='LunarLander-v2')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--max_timesteps", type=int, default=int(1.5e+5))
    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.01)
    parser.add_argument("--eps_decay", type=float, default=0.995)

    args = parser.parse_args()
    return args


def test(env):
    agent = DoubleDQNAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        env_name=env.unwrapped.spec.id,
    )
    agent.load()

    obs = env.reset()
    score = 0
    while True:
        action = agent.act(obs, train_mode=False)
        next_obs, reward, done, _ = env.step(action)
        env.render()

        obs = next_obs
        score += reward

        if done:
            print(f'ep score: {score:.2f}')
            obs = env.reset()
            score = 0


def train(env, args):
    agent = DoubleDQNAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        env_name=env.unwrapped.spec.id,
        buffer_size=args.buffer_size,
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
    )

    if args.cont:
        agent.load()

    start = time()

    obs = env.reset()
    score = 0
    for t in range(1, args.max_timesteps+1):
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)

        agent.step(obs, action, reward, next_obs, done)
        obs = next_obs
        score += reward

        if done:
            print(f'{t}/{args.max_timesteps} | ep score: {score:.2f}')
            obs = env.reset()
            score = 0

        if t % (args.max_timesteps // 10) == 0:
            agent.save()

    end = time()
    print("training completed, elapsed time: ", end - start)

    agent.save()


def main():
    args = get_args()
    env = gym.make(args.env_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    if args.test:
        test(env)
    else:
        train(env, args)


if __name__ == "__main__":
    main()
