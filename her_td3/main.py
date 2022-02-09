import random
import gym
import numpy as np
import json
import torch
import argparse
from collections import deque
from her_td3_agent import HER_TD3Agent
from time import time


def read_hyperparams():
    with open('her_td3/hyperparams.json') as f:
        hyperparams = json.load(f)
        return hyperparams


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='FetchReach-v1')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=2)

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
        train_mode=False
    )
    agent.load()

    scores = deque(maxlen=10)
    for i in range(1, 1000):
        env_dict = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(env_dict["observation"], env_dict["desired_goal"])
            next_env_dict, reward, done, _ = env.step(action)
            env.render()

            env_dict = next_env_dict
            score += reward

        scores.append(score)
        mean_score = np.mean(scores)

        print(f'\rEpisode: {i} \tAverage Score: {mean_score:.2f}', end="")
        if i % 10 == 0:
            print(f'\rEpisode: {i} \tAverage Score: {mean_score:.2f}')


def train(env):
    hyperparams = read_hyperparams()

    agent = HER_TD3Agent(
        obs_dim=env.observation_space["observation"].shape[0],
        action_dim=env.action_space.shape[0],
        goal_dim=env.observation_space["desired_goal"].shape[0],
        action_bounds={"low": env.action_space.low, "high": env.action_space.high},
        compute_reward_func=env.compute_reward,
        k_future=hyperparams["k_future"],
        env_name=env.unwrapped.spec.id,
        buffer_size=hyperparams['buffer_size'],
        actor_lr=hyperparams['actor_lr'],
        critic_lr=hyperparams['critic_lr'],
        batch_size=hyperparams['batch_size'],
        gamma=hyperparams['gamma'],
        tau=hyperparams['tau'],
        policy_noise=hyperparams['policy_noise'],
        noise_clip=hyperparams['noise_clip'],
        policy_freq=hyperparams['policy_freq'],
        train_mode=True
    )

    start = time()

    max_episodes = hyperparams['max_episodes']
    scores = deque(maxlen=10)
    for i in range(1, max_episodes+1):
        env_dict = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(env_dict["observation"], env_dict["desired_goal"])
            next_env_dict, reward, done, _ = env.step(action)

            agent.step(env_dict["observation"], action, reward, next_env_dict["observation"],
                       env_dict["desired_goal"], next_env_dict["achieved_goal"], done)
            env_dict = next_env_dict
            score += reward

        scores.append(score)
        mean_score = np.mean(scores)

        print(f'\rEpisode: {i}/{max_episodes} \tAverage Score: {mean_score:.2f}', end="")
        if i % 10 == 0:
            print(f'\rEpisode: {i}/{max_episodes} \tAverage Score: {mean_score:.2f}')

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

    if args.test:
        test(env)
    else:
        train(env)


if __name__ == "__main__":
    main()
