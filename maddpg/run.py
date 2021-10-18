from pettingzoo.mpe import simple_adversary_v2
from maddpg import MADDPG
import sys
from time import sleep

load_models = len(sys.argv) >= 2 and sys.argv[1] == "load"
eval_mode = len(sys.argv) >= 3 and sys.argv[1] == "load" and sys.argv[2] == "eval"

env = simple_adversary_v2.parallel_env(continuous_actions=True)
maddpg = MADDPG(env.possible_agents, env.observation_spaces, env.action_spaces, load_models)
n_episodes = 100000

print("MODE: ", "eval" if eval_mode else "train")
for i in range(1, n_episodes):
    observations = env.reset()
    dones = {agent: False for agent in env.possible_agents}

    while not any(dones.values()):
        actions = maddpg.act(observations, eval_mode)
        next_observations, rewards, dones, _ = env.step(actions)

        if eval_mode:
            env.render()
            sleep(0.1)
        else:
            maddpg.step(observations, actions, rewards, next_observations, dones)

        observations = next_observations

    if i % 100 == 0 and not eval_mode:
        print("episode: ", i)
        maddpg.save_models()
