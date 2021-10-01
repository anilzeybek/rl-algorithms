import ray
from model import QNetwork
import gym


@ray.remote
class APEXActor:
    def init(self, num, env_name, args):
        self.num = num
        self.q_network = QNetwork(args['state_size'], args['action_size'])
        self.env = gym.make(env_name)
        self.args = args

    def _act(self, state):
        if np.random.rand() < self.args['eps']:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).unsqueeze(0).float()
            action_values = self.q_network(state)
            return torch.argmax(action_values).item()

    def _run_env(self):
        if env.can_run:
            state = env.state
        else:
            state = env.reset()

        trajectories = []
        for t in range(env._max_episode_steps):
            action = self._act(torch.from_numpy(state).float())

            next_state, reward, done, _ = env.step(action)
            trajectories.append([state, action, reward, next_state, done])

            if done:
                state = env.reset()
            else:
                state = next_state

        return trajectories

    def run(self, ps, global_buffer):
        while True:
            if i % self.args['actor_update_cycle'] == 0:
                weights = ray.get(ps.pull.remote())
                self.q_network.load_state_dict(weights)

            trajectories = _run_env()
            global_buffer.store_transition.remote(trajectories)
