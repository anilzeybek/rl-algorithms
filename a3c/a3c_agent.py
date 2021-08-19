import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from model import PolicyNetwork, VNetwork


class A3CAgent(mp.Process):
    def __init__(self, env, global_critic_network, global_actor_network, critic_optimizer, actor_optimizer, state_size, action_size, global_ep_idx, n_episodes):
        super(A3CAgent, self).__init__()

        self.local_critic_network = VNetwork(state_size)
        self.global_critic_network = global_critic_network

        self.local_actor_network = PolicyNetwork(state_size, action_size)
        self.global_actor_network = global_actor_network

        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer

        self.env = env
        self.global_ep_idx = global_ep_idx
        self.n_episodes = n_episodes
        self.gamma = 0.99

    def act(self, state):
        state = torch.from_numpy(state).unsqueeze(0).float()
        probabilities = F.softmax(self.local_actor_network(state), dim=1)

        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()

        self.log_prob = action_probs.log_prob(action)
        return action.item()

    def run(self):
        while self.global_ep_idx.value < self.n_episodes:
            state = self.env.reset()
            score = 0
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self._step(state, action, reward, next_state, done)

                state = next_state
                score += reward

            with self.global_ep_idx.get_lock():
                self.global_ep_idx.value += 1

            print('episode ', self.global_ep_idx.value, ' score %.1f' % score)

    def _step(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).unsqueeze(0).float()
        next_state = torch.from_numpy(next_state).unsqueeze(0).float()

        v_current = self.local_critic_network(state)
        with torch.no_grad():
            v_target = reward + self.gamma * self.local_critic_network(next_state) * (1 - int(done))

        self.critic_optimizer.zero_grad()
        critic_loss = (v_target - v_current)**2
        critic_loss.backward()
        for local_param, global_param in zip(self.local_critic_network.parameters(), self.global_critic_network.parameters()):
            global_param._grad = local_param.grad
        self.critic_optimizer.step()

        advantage = (v_target - v_current).detach()
        self.actor_optimizer.zero_grad()
        actor_loss = -(advantage * self.log_prob)
        actor_loss.backward()
        for local_param, global_param in zip(self.local_actor_network.parameters(), self.global_actor_network.parameters()):
            global_param._grad = local_param.grad
        self.actor_optimizer.step()

        self.local_critic_network.load_state_dict(self.global_critic_network.state_dict())
        self.local_actor_network.load_state_dict(self.global_actor_network.state_dict())
