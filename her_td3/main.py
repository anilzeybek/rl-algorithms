import argparse
import random

import gym
import numpy as np
import torch

from her_td3_agent import HER_TD3Agent

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from common.utils import test, try_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='FetchPush-v1')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--max_timesteps", type=int, default=int(5e+5))
    parser.add_argument("--expl_noise", type=float, default=0.1)
    parser.add_argument("--start_timesteps", type=int, default=25000)
    parser.add_argument("--k_future", type=int, default=4)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)

    args = parser.parse_args()
    return args


def train(env, agent, args):
    if args.cont:
        agent.load()

    env_dict = env.reset(seed=args.seed)
    score = 0
    last_checkpoint_at = 0
    best_eval_score = -9999
    for t in range(1, args.max_timesteps + 1):
        action = agent.act(env_dict["observation"], env_dict["desired_goal"])
        next_env_dict, reward, done, _ = env.step(action)

        agent.step(env_dict, action, reward, next_env_dict, done)
        env_dict = next_env_dict
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

    agent = HER_TD3Agent(
        obs_dim=env.observation_space["observation"].shape[0],
        action_dim=env.action_space.shape[0],
        goal_dim=env.observation_space["desired_goal"].shape[0],
        action_bounds={"low": env.action_space.low, "high": env.action_space.high},
        compute_reward_func=env.compute_reward,
        env_name=args.env_name,
        k_future=args.k_future,
        expl_noise=args.expl_noise,
        start_timesteps=args.start_timesteps,
        buffer_size=args.buffer_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq
    )

    if args.test:
        test(env, agent)
    else:
        train(env, agent, args)


if __name__ == "__main__":
    main()
