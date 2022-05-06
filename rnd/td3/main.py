import argparse
import random
from time import time

import gym
import numpy as np
import torch

from td3_agent import TD3Agent


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--max_timesteps", type=int, default=int(1.5e+5))
    parser.add_argument("--expl_noise", type=float, default=0.1)
    parser.add_argument("--start_timesteps", type=int, default=25000)
    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--predictor_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--initial_normalization_episodes", type=int, default=100)

    args = parser.parse_args()
    return args


def test(env, agent):
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


def train(env, agent, args):
    if args.cont:
        agent.load()

    start = time()

    obs_list = []
    for _ in range(args.initial_normalization_episodes):
        obs = env.reset()
        obs_list.append(obs)
        done = False
        while not done:
            obs, _, done, _ = env.step(env.action_space.sample())
            obs_list.append(obs)

    agent.obs_normalizer.update(np.array(obs_list))

    obs = env.reset()
    score = 0
    for t in range(1, args.max_timesteps+1):
        action = agent.act(obs)
        next_obs, ext_reward, done, _ = env.step(action)

        agent.step(obs, action, ext_reward, next_obs, done)
        obs = next_obs
        score += ext_reward

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

    agent = TD3Agent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_bounds={"low": env.action_space.low, "high": env.action_space.high},
        env_name=env.unwrapped.spec.id,
        expl_noise=args.expl_noise,
        start_timesteps=args.start_timesteps,
        buffer_size=args.buffer_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        predictor_lr=args.predictor_lr,
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
