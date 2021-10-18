import numpy as np
import torch
from ma_replay_buffer import MAReplayBuffer
from agent import Agent


BUFFER_SIZE = 100000
BATCH_SIZE = 64
START_STEPS = 5000
UPDATE_EVERY = 10
UPDATE_AFTER = 1000


class MADDPG:
    def __init__(self, n_agents, obs_dims, n_actions):
        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.n_actions = n_actions
        self.t = 0

        self.replay_buffer = MAReplayBuffer(BUFFER_SIZE, n_agents)
        self.agents = [Agent(i, n_agents, sum(obs_dims), obs_dims[i], n_actions) for i in range(n_agents)]

    def act(self, agent_id, obs):
        if self.t > START_STEPS:
            action = self.agents[agent_id].choose_action(obs)
        else:
            action = np.random.randint(self.n_actions)

        return action

    def step(self, observations, actions, rewards, next_observations, dones):
        self.t += 1

        self.replay_buffer.store_transition(observations, actions, rewards, next_observations, dones)
        if self.t >= UPDATE_AFTER and self.t % UPDATE_EVERY == 0:
            for _ in range(UPDATE_EVERY):
                batch = self.replay_buffer.sample(BATCH_SIZE)
                self._learn(batch)

    def _learn(self, data):
        observations, actions, rewards, next_observations, dones = data
        state = torch.cat(observations, axis=1)
        next_states = torch.cat(next_observations, axis=1)
        actions = torch.cat(actions, axis=1)

        all_agents_target_actor = []
        for i, agent in enumerate(self.agents):
            all_agents_target_actor.append(agent.actor_target(next_observations[i]))
        all_agents_target_actor = torch.cat(all_agents_target_actor, axis=1)

        for i, agent in enumerate(self.agents):
            agent.critic_optimizer.zero_grad()
            loss_Q = agent.compute_loss_q(state, actions, rewards[i], next_states, dones[i], all_agents_target_actor)
            loss_Q.backward()
            agent.critic_optimizer.step()

        # # Freeze Q-network so you don't waste computational effort
        # # computing gradients for it during the policy learning step.
        for agent in self.agents:
            for p in agent.critic_network.parameters():
                p.requires_grad = False

        for agent in self.agents:
            all_agents_actor = []
            for i, agent_ in enumerate(self.agents):
                all_agents_actor.append(agent_.actor_network(observations[i]))
            all_agents_actor = torch.cat(all_agents_actor, axis=1)

            agent.actor_optimizer.zero_grad()
            loss_pi = agent.compute_loss_pi(state, all_agents_actor)
            loss_pi.backward()
            agent.actor_optimizer.step()

        # # Unfreeze Q-network so you can optimize it at next DDPG step.
        for agent in self.agents:
            for p in agent.critic_network.parameters():
                p.requires_grad = True

        for agent in self.agents:
            agent.update_network_parameters()
