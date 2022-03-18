import argparse
import json
import random
from time import time

import gym
import numpy as np
import torch

from rnd_ppo_agent import RND_PPOAgent


def read_hyperparams():
    with open('rnd/hyperparams.json') as f:
        hyperparams = json.load(f)
        return hyperparams


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='LunarLander-v2')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return args


def test(env):
    agent = RND_PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        env_name=env.unwrapped.spec.id,
    )
    agent.load()

    for _ in range(1, 2500):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            env.render()

            obs = next_obs
            score += reward

        print(f"score: {score:.2f}")


def train(env, cont):
    hyperparams = read_hyperparams()

    agent = RND_PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        env_name=env.unwrapped.spec.id,
        actor_lr=hyperparams['actor_lr'],
        critic_lr=hyperparams['critic_lr'],
        predictor_lr=hyperparams['predictor_lr'],
        gamma=hyperparams['gamma'],
        gae_lambda=hyperparams['gae_lambda'],
        clip_ratio=hyperparams['clip_ratio'],
        target_kl=hyperparams['target_kl'],
        train_actor_iters=hyperparams['train_actor_iters'],
        train_critic_iters=hyperparams['train_critic_iters'],
        train_predictor_iters=hyperparams['train_predictor_iters'],
    )

    if cont:
        agent.load()

    start = time()

    obs_list = []
    for _ in range(hyperparams['initial_normalization_episodes']):
        obs = env.reset()
        obs_list.append(obs)
        done = False
        while not done:
            obs, _, done, _ = env.step(env.action_space.sample())
            obs_list.append(obs)

    agent.obs_normalizer.update(np.array(obs_list))

    max_episodes = hyperparams['max_episodes']
    for i in range(1, max_episodes+1):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, extrinsic_reward, done, _ = env.step(action)

            agent.step(obs, action, extrinsic_reward, next_obs, done)
            obs = next_obs
            score += extrinsic_reward

        if i % 100 == 0:
            agent.save()

        print(f'ep: {i}/{max_episodes} | score: {score:.2f}')

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
        train(env, args.cont)


if __name__ == "__main__":
    main()
