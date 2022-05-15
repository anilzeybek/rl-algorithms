import argparse
import random
from time import time

import gym
import numpy as np
import torch

from rnd_ppo_agent import RND_PPOAgent


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='MountainCar-v0')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--max_timesteps", type=int, default=int(3e+4))
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--predictor_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.97)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--train_actor_iters", type=int, default=80)
    parser.add_argument("--train_critic_iters", type=int, default=80)
    parser.add_argument("--train_predictor_iters", type=int, default=80)
    parser.add_argument("--initial_normalization_episodes", type=int, default=100)

    args = parser.parse_args()
    return args


def test(env, agent):
    agent.load()

    obs = env.reset()
    score = 0
    while True:
        action = agent.act(obs)
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

    agent = RND_PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        env_name=args.env_name,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        predictor_lr=args.predictor_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
        train_actor_iters=args.train_actor_iters,
        train_critic_iters=args.train_critic_iters,
        train_predictor_iters=args.train_predictor_iters
    )

    if args.test:
        test(env, agent)
    else:
        train(env, agent, args)


if __name__ == "__main__":
    main()
