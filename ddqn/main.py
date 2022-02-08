import random
import gym
import numpy as np
import json
import torch
import argparse
from collections import deque
from ddqn_agent import DDQNAgent


def read_hyperparams():
    with open('ddqn/hyperparams.json') as f:
        hyperparams = json.load(f)
        return hyperparams


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='LunarLander-v2')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return args


def test(env):
    agent = DDQNAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        env_name=env.unwrapped.spec.id,
        train_mode=False
    )
    agent.load()

    scores = deque(maxlen=10)
    for i in range(1, 1000):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            env.render()

            obs = next_obs
            score += reward

        scores.append(score)
        mean_score = np.mean(scores)

        print(f'\rEpisode: {i} \tAverage Score: {mean_score:.2f}', end="")
        if i % 10 == 0:
            print(f'\rEpisode: {i} \tAverage Score: {mean_score:.2f}')


def train(env):
    hyperparams = read_hyperparams()

    agent = DDQNAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        env_name=env.unwrapped.spec.id,
        buffer_size=hyperparams['buffer_size'],
        lr=hyperparams['lr'],
        batch_size=hyperparams['batch_size'],
        gamma=hyperparams['gamma'],
        tau=hyperparams['tau'],
        eps_start=hyperparams['eps_start'],
        eps_end=hyperparams['eps_end'],
        eps_decay=hyperparams['eps_decay'],
        train_mode=True
    )

    max_episodes = hyperparams['max_episodes']
    scores = deque(maxlen=10)
    for i in range(1, max_episodes+1):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)

            agent.step(obs, action, reward, next_obs, done)
            obs = next_obs
            score += reward

        scores.append(score)
        mean_score = np.mean(scores)

        print(f'\rEpisode: {i}/{max_episodes} \tAverage Score: {mean_score:.2f}', end="")
        if i % 10 == 0:
            print(f'\rEpisode: {i}/{max_episodes} \tAverage Score: {mean_score:.2f}')

    agent.save()


def main():
    args = get_args()
    env = gym.make(args.env_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    if args.test:
        test(env)
    else:
        train(env)


if __name__ == "__main__":
    main()
