import gym
import numpy as np
from ddpg_agent import DDPGAgent
import random
import torch

N_EPISODES = 500


def eval(env: gym.Env, agent: DDPGAgent) -> float:
    scores = []

    for _ in range(3):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.act(state, train_mode=False)
            next_state, reward, done, _ = env.step(action)

            state = next_state
            score += reward

        scores.append(score)
    return np.mean(scores)


def main():
    env = gym.make('Pendulum-v1')

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    env.seed(0)
    torch.manual_seed(seed)

    agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low[0], env.action_space.high[0])
    for i in range(1, N_EPISODES+1):
        state = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.step(state, action, reward, next_state, done)
            state = next_state

        print(f'\rEpisode: {i}', end="")
        if i % 10 == 0:
            print(f'\rEpisode: {i}\tEval score: {eval(env, agent):.2f}')

    while True:
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.act(state, train_mode=False)
            next_state, reward, done, _ = env.step(action)
            env.render()

            state = next_state
            score += reward

        print(score)


if __name__ == "__main__":
    main()
