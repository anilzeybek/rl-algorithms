import random
import gym
import numpy as np
import json
import torch
import argparse
from her_td3_agent import HER_TD3Agent
from time import time


def read_hyperparams():
    with open('her_td3/hyperparams.json') as f:
        hyperparams = json.load(f)
        return hyperparams


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='FetchPush-v1')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--use_saved', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)

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


def train(env, use_saved):
    hyperparams = read_hyperparams()

    agent = HER_TD3Agent(
        obs_dim=env.observation_space["observation"].shape[0],
        action_dim=env.action_space.shape[0],
        goal_dim=env.observation_space["desired_goal"].shape[0],
        action_bounds={"low": env.action_space.low, "high": env.action_space.high},
        compute_reward_func=env.compute_reward,
        k_future=hyperparams["k_future"],
        env_name=env.unwrapped.spec.id,
        expl_noise=hyperparams['expl_noise'],
        start_timesteps=hyperparams['start_timesteps'],
        buffer_size=hyperparams['buffer_size'],
        actor_lr=hyperparams['actor_lr'],
        critic_lr=hyperparams['critic_lr'],
        batch_size=hyperparams['batch_size'],
        gamma=hyperparams['gamma'],
        tau=hyperparams['tau'],
        policy_noise=hyperparams['policy_noise'],
        noise_clip=hyperparams['noise_clip'],
        policy_freq=hyperparams['policy_freq'],
        use_saved=use_saved
    )

    start = time()

    max_episodes = hyperparams['max_episodes']
    for i in range(1, max_episodes+1):
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
        train(env, args.use_saved)


if __name__ == "__main__":
    main()
