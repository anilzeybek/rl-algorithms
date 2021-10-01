import ray
import torch.optim as optim
from copy import deepcopy
from model import QNetwork


@ray.remote
class APEXLearner:
    def init(self, args):
        self.q_network = QNetwork(args['state_size'], args['action_size'])
        self.target_network = deepcopy(self.q_network)
        self.args = args

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.args['lr'])
        self.update_num = 0

    def _get_td_error(self, data):
        states, actions, rewards, next_states, dones = data

        Q_current = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            Q_targets_next = self.target_network(next_states).max(1)[0].unsqueeze(1)
            Q_targets = rewards + self.args['gamma'] * Q_targets_next * (1 - dones)

        loss = F.mse_loss(Q_current, Q_targets)
        return loss

    def _train_network(self, data):
        td_error = self._get_td_error(data)

        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

        self.update_num += 1
        if self.update_num % self.args['target_update_cycle'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def run(self, ps, buffer):
        data = ray.get(buffer.sample.remote(self.args['batch_size']))
        self.model._train_network(data)
        ray.wait([ps.push.remote(self.get_weights())])
