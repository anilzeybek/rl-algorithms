import argparse
import os
import random

import gym
import numpy as np
import torch

from sac_agent import SACAgent


import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from common.utils import train, test


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v3')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--max_timesteps", type=int, default=int(1.5e+5))
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--start_timesteps", type=int, default=25000)
    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    env = gym.make(args.env_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.action_space.seed(args.seed)

    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_bounds={"low": env.action_space.low, "high": env.action_space.high},
        env_name=args.env_name,
        alpha=args.alpha,
        start_timesteps=args.start_timesteps,
        buffer_size=args.buffer_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau
    )

    if args.test:
        test(env, agent)
    else:
        train(env, agent, args)


if __name__ == "__main__":
    main()
