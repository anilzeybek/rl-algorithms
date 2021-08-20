import gym
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from model import PolicyNetwork, VNetwork
from a3c_agent import A3CAgent


N_EPISODES = 50000
LR = 1e-4

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    global_critic_network = VNetwork(state_size)
    global_critic_network.share_memory()

    global_actor_network = PolicyNetwork(state_size, action_size)
    global_actor_network.share_memory()

    critic_optimizer = SharedAdam(global_critic_network.parameters(), lr=LR)
    actor_optimizer = SharedAdam(global_actor_network.parameters(), lr=LR)

    global_ep_idx = mp.Value('i', 0)

    workers = [A3CAgent(env, global_critic_network, global_actor_network, critic_optimizer, actor_optimizer, state_size, action_size, global_ep_idx, N_EPISODES) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]
