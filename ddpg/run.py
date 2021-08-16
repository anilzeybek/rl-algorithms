import gym
import numpy as np
import sys
from collections import deque
from ddpg_agent import DDPGAgent

N_EPISODES = 200
env = gym.make('MountainCarContinuous-v0')
agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high)

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

    if i > 50 and score > 75:
        break

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
        env.render()
        
        state = next_state
        score += reward

    print(score)
