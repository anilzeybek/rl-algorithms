import ray
import torch
import time

from learner import APEXLearner
from actor import APEXActor
from replay_buffer import ApexBuffer
from parameter_server import ParameterServer

NUM_ACTORS = 4
BUFFER_SIZE = 100000
ENV_NAME = 'LunarLander-v2'

env = gym.make(ENV_NAME)
learner_args = {
    'state_size': env.observation_space.shape[0],
    'action_size': env.action_space.n,
    'gamma': 0.99,
    'target_update_cycle': pass,
    'batch_size': pass
}

actor_args = {
    'state_size': env.observation_space.shape[0],
    'action_size': env.action_space.n,
    'eps': pass,
    'actor_update_cycle': pass
}

learner = APEXLearner.remote()
actors = [APEXActor.remote() for _ in range(NUM_ACTORS)]
buffer = ApexBuffer.remote(BUFFER_SIZE)

learner.init.remote(learner_args)
ray.get([agent.init.remote(idx, actor_args) for idx, agent in enumerate(actors)])
ps = ParameterServer.remote(ray.get(learner.get_weights.remote()))

[actor.run.remote(args.env_name, ps, buffer, args.epochs) for actor in actors]
time.sleep(3)
print('learner start')
for epoch in range(args.epochs):
    ray.wait([learner.run.remote(ps, buffer)])
print('learner finish')
