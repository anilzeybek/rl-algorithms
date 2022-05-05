import argparse
import random
from time import time

import gym
import numpy as np
import torch

from her_td3_agent import HER_TD3Agent


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='FetchPush-v1')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--max_episodes", type=int, default=10000)
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


def test(env):
    agent = HER_TD3Agent(
        obs_dim=env.observation_space["observation"].shape[0],
        action_dim=env.action_space.shape[0],
        goal_dim=env.observation_space["desired_goal"].shape[0],
        action_bounds={"low": env.action_space.low, "high": env.action_space.high},
        compute_reward_func=env.compute_reward,
        env_name=env.unwrapped.spec.id,
    )
    agent.load()

    for _ in range(1, 1000):
        env_dict = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(env_dict["observation"], env_dict["desired_goal"], train_mode=False)
            next_env_dict, reward, done, _ = env.step(action)
            env.render()

            env_dict = next_env_dict
            score += reward

        print(f"score: {score:.2f}")


def train(env, args):
    agent = HER_TD3Agent(
        obs_dim=env.observation_space["observation"].shape[0],
        action_dim=env.action_space.shape[0],
        goal_dim=env.observation_space["desired_goal"].shape[0],
        action_bounds={"low": env.action_space.low, "high": env.action_space.high},
        compute_reward_func=env.compute_reward,
        env_name=env.unwrapped.spec.id,
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

    if args.cont:
        agent.load()

    start = time()

    for i in range(1, args.max_episodes+1):
        env_dict = env.reset()
        while np.linalg.norm(env_dict["achieved_goal"] - env_dict["desired_goal"]) <= 0.05:
            env_dict = env.reset()

        score = 0
        done = False
        while not done:
            action = agent.act(env_dict["observation"], env_dict["desired_goal"])
            next_env_dict, reward, done, _ = env.step(action)

            agent.step(env_dict, action, reward, next_env_dict, done)
            env_dict = next_env_dict
            score += reward

        if i % 100 == 0:
            agent.save()

        print(f'ep: {i}/{args.max_episodes} | score: {score:.2f}')

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
