from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
from collections import deque
from pyquaternion import Quaternion
from ddpg_her_agent import DDPG_HERAgent


obs_config = ObservationConfig()
#obs_config.set_all(True)
obs_config.set_all_high_dim(False)
obs_config.set_all_low_dim(True)

action_mode = ActionMode(ArmActionMode.EE_POSE_EE_FRAME)
env = Environment(action_mode, obs_config=obs_config, headless=False)
env.launch()

task = env.get_task(ReachTarget)
agent = DDPG_HERAgent(6, 3, 0.01)

num_episodes = 100000
episode_length = 1000

scores = deque(maxlen=10)
for i in range(1, num_episodes+1):
    _, obs = task.reset()
    eef = obs.gripper_pose[:3]
    
    q = Quaternion(axis=(0, 1, 0), degrees=0)
    eef_ori = np.array([q.x, q.y, q.z, q.w])
    gripper_state = obs.gripper_open
    
    target = task._task.target.get_position()
    obs = np.concatenate((eef, target))

    score = 0
    done = False
    episode_count = 0
    while not done:
        episode_count += 1

        action = agent.act(obs, noise=0.005)
        action = np.concatenate((action, eef_ori, [gripper_state]))
        try:
            next_obs, reward, done = task.step(action)
        except:
            if episode_count >= episode_length:
                break
            else:
                continue

        if episode_count >= episode_length:
            done = True

        eef = next_obs.gripper_pose[:3]
        next_obs = np.concatenate((eef, target))
        
        agent.step(obs, action[:3], reward, next_obs, done)
        obs = next_obs
        score += reward

    scores.append(score)
    mean_score = np.mean(scores)
    print(f'\rEpisode: {i}\tAverage Score: {mean_score:.2f}', end="")
    if i % 10 == 0:
        print(f'\rEpisode: {i}\tAverage Score: {mean_score:.2f}')
    
