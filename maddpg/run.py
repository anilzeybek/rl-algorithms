from pettingzoo.mpe import simple_adversary_v2
from maddpg import MADDPG


env = simple_adversary_v2.env()
env.reset()

n_agents = env.num_agents
obs_dims = [env.observation_spaces[key].shape[0] for key in env.observation_spaces]
maddpg = MADDPG(n_agents, obs_dims, 5)


for i in range(50000):
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for agent_id in range(n_agents):
        obs, reward, done, _ = env.last()
        action = maddpg.act(agent_id, obs)

        observations.append(obs)
        actions.append(action)
        if i > 0:
            rewards.append(reward)
            next_observations.append(obs)
            dones.append(done)

        if done:
            env.step(None)
            continue

        env.step(action)
        if i > 5000:
            env.render()

    if any(dones):
        env.reset()

    if i > 0:
        maddpg.step(observations, actions, rewards, next_observations, dones)

    print(f"{i}: {sum(rewards):.2f}")