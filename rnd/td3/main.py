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

    parser.add_argument("--max_timesteps", type=int, default=int(1.5e+4))
    parser.add_argument("--expl_noise", type=float, default=0.1)
    parser.add_argument("--start_timesteps", type=int, default=5000)
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
    parser.add_argument("--initial_normalization_timesteps", type=int, default=100000)

    args = parser.parse_args()
    return args


def eval_agent(env, agent, times=1, print_score=False, render=False):
    scores = []

    for _ in range(times):
        obs = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.act(obs, train_mode=False)
            next_obs, reward, done, _ = env.step(action)
            if render:
                env.render()

            obs = next_obs
            score += reward

        scores.append(score)
        if print_score:
            print(score)

    return sum(scores) / len(scores)


def test(env, agent):
    agent.load()

    score = eval_agent(env, agent, print_score=True, times=50)
    print(f"avt score: {score:.2f}")


def try_checkpoint(env, agent, best_eval_score):
    current_eval_score = eval_agent(env, agent, times=20)
    if current_eval_score > best_eval_score:
        best_eval_score = current_eval_score
        print(f"checkpoint eval_score={current_eval_score:.2f}")
        agent.save()

    return best_eval_score


def train(env, agent, args):
    if args.cont:
        agent.load()

    c = 0

    obs_list = []
    obs = env.reset(seed=args.seed)
    for t in range(args.initial_normalization_timesteps):
        obs_list.append(obs)
        obs, _, done, _ = env.step(env.action_space.sample())
        if done:
            c += 1
            obs = env.reset()

    print(c)

    agent.obs_normalizer.update(np.array(obs_list))

    obs = env.reset()
    score = 0
    last_checkpoint_at = 0
    best_eval_score = -9999
    for t in range(1, args.max_timesteps+1):
        action = agent.act(obs)
        next_obs, ext_reward, done, _ = env.step(action)

        agent.step(obs, action, ext_reward, next_obs, done)
        obs = next_obs
        score += ext_reward

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

    agent = TD3Agent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_bounds={"low": env.action_space.low, "high": env.action_space.high},
        env_name=args.env_name,
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
