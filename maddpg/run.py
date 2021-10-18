from pettingzoo.mpe import simple_adversary_v2
from maddpg import MADDPG

env = simple_adversary_v2.parallel_env()
maddpg = MADDPG(env.possible_agents, env.observation_spaces, 5)

for i in range(100000):
    observations = env.reset()

    dones = [False for _ in env.agents]
    while not any(dones):
        actions = maddpg.act(observations)
        next_observations, rewards, dones, _ = env.step(actions)

        maddpg.step(observations, actions, rewards, next_observations, dones)
        observations = next_observations
