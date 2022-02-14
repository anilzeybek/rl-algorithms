import random
import gym
import numpy as np
import json
import torch
import argparse
from collections import deque
from dueling_dqn_agent import DuelingDQNAgent
from time import time


def read_hyperparams():
    with open('dueling_dqn/hyperparams.json') as f:
        hyperparams = json.load(f)
        return hyperparams


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return args


def test(env):
    agent = DuelingDQNAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        env_name=env.unwrapped.spec.id,
    )
    agent.load()

    for _ in range(1, 1000):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(obs, train_mode=False)
            next_obs, reward, done, _ = env.step(action)
            env.render()

            obs = next_obs
            score += reward

        print(score)


def train(env):
    hyperparams = read_hyperparams()

    agent = DuelingDQNAgent(
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
    )

    start = time()

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

        print(f'ep: {i}/{max_episodes} | score: {mean_score:.2f}')

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
        train(env)


if __name__ == "__main__":
    main()
