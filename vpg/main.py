import argparse
import os
import random

import gym
import numpy as np
import torch

from vpg_agent import VPGAgent


import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from common.utils import train, test


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='LunarLander-v2')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--max_timesteps", type=int, default=int(6e+5))
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    env = gym.make(args.env_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.action_space.seed(args.seed)

    agent = VPGAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        env_name=args.env_name,
        actor_lr=args.actor_lr,
        gamma=args.gamma,
    )

    if args.test:
        test(env, agent)
    else:
        train(env, agent, args)


if __name__ == "__main__":
    main()
