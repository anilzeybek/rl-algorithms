import argparse
import os
import random

import gym
import numpy as np
import torch

from dueling_dqn_agent import DuelingDQNAgent

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from common.utils import test, try_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='LunarLander-v2')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--max_timesteps", type=int, default=int(1.5e+5))
    parser.add_argument("--start_timesteps", type=int, default=1000)
    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--eps_init", type=float, default=1.0)
    parser.add_argument("--eps_last", type=float, default=0.05)
    parser.add_argument("--eps_end_fraction", type=float, default=0.4)

    args = parser.parse_args()
    return args


def train(env, agent, args):
    if args.cont:
        agent.load()

    obs = env.reset(seed=args.seed)
    score = 0
    last_checkpoint_at = 0
    best_eval_score = -9999
    for t in range(1, args.max_timesteps + 1):
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)

        agent.step(obs, action, reward, next_obs, done)
        obs = next_obs
        score += reward

        if done:
            print(f'{t}/{args.max_timesteps} | ep score: {score:.2f}')

            if t - last_checkpoint_at > (args.max_timesteps // 10):
                best_eval_score = try_checkpoint(env, agent, best_eval_score)
                last_checkpoint_at = t

            score = 0
            obs = env.reset()

    try_checkpoint(env, agent, best_eval_score)


def main():
    args = get_args()
    env = gym.make(args.env_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.action_space.seed(args.seed)

    agent = DuelingDQNAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        env_name=args.env_name,
        start_timesteps=args.start_timesteps,
        buffer_size=args.buffer_size,
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        eps_init=args.eps_init,
        eps_last=args.eps_last,
        eps_decay=(args.eps_init - args.eps_last) / (args.max_timesteps * args.eps_end_fraction),
    )

    if args.test:
        test(env, agent)
    else:
        train(env, agent, args)


if __name__ == "__main__":
    main()
