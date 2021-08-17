import gym
import numpy as np
import sys
from collections import deque
from ddpg_agent import DDPGAgent
from easy_env.easy_env import EasyEnv

N_EPISODES = 100000
env = EasyEnv(length=50)
agent = DDPGAgent(2, 1, 1)

scores = deque(maxlen=10)
for i in range(1, N_EPISODES+1):
    state = env.reset()
    score = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

    scores.append(score)
    mean_score = np.mean(scores)

    print(f'\rEpisode: {i}\tAverage Score: {mean_score:.2f}', end="")
    if i % 10 == 0:
        print(f'\rEpisode: {i}\tAverage Score: {mean_score:.2f}')


while True:
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        action = agent.act(state, noise=0)
        next_state, reward, done, _ = env.step(action)
        
        state = next_state
        score += reward

    print(score)
