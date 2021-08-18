import gym
import numpy as np
import sys
from collections import deque
from ddpg_her_agent import DDPG_HERAgent

N_EPISODES = 100000
env = gym.make('FetchReach-v1')
obs_space = env.observation_space['achieved_goal'].shape[0] + env.observation_space['desired_goal'].shape[0] 
agent = DDPG_HERAgent(obs_space, env.action_space.shape[0], env.action_space.high)

scores = deque(maxlen=10)
for i in range(1, N_EPISODES+1):
    state = env.reset()
    state = np.concatenate((state['achieved_goal'], state['desired_goal']))

    score = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.concatenate((next_state['achieved_goal'], next_state['desired_goal']))

        env.render()

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
        env.render()
        
        state = next_state
        score += reward

    print(score)
