from comet_ml import Experiment
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
from collections import deque
from pyquaternion import Quaternion
from ddpg_her_agent import DDPG_HERAgent


def get_experiment(exp_name, exp_tags):
    experiment = Experiment(
        api_key="4iMhvGHg82QsAVYdkRfxTzesW",
        project_name="rl",
        workspace="anilz",
        auto_output_logging="simple"
    )

    experiment.set_name(exp_name)
    experiment.add_tags(exp_tags)

    return experiment


if __name__ == "__main__":
    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(True)

    action_mode = ActionMode(ArmActionMode.EE_POSE_EE_FRAME)
    env = Environment(action_mode, obs_config=obs_config, headless=False)
    env.launch()

    task = env.get_task(ReachTarget)
    experiment = get_experiment("RLBench Reacher", ["DDPG", "HER", "EE_POSE"])
    agent = DDPG_HERAgent(6, 3, 0.02, experiment)

    num_episodes = 100000
    max_episode_length = 2000
    action_noise = 0.02

    experiment.log_parameter("max_episode_length", max_episode_length)
    experiment.log_parameter("action_noise", action_noise)

    scores = deque(maxlen=10)
    for i in range(1, num_episodes+1):
        _, obs = task.reset()
        eef = obs.gripper_pose[:3]
        
        q = Quaternion(axis=(1, 0, 0), degrees=0)
        eef_ori = np.array([q.x, q.y, q.z, q.w])
        gripper_state = obs.gripper_open
        
        target = task._task.target.get_position()
        obs = np.concatenate((eef, target))

        score = 0
        done = False
        episode_length = 0
        while not done:
            episode_length += 1

            action = agent.act(obs, noise=action_noise)
            action = np.concatenate((action, eef_ori, [gripper_state]))
            try:
                next_obs, reward, done = task.step(action)
            except:
                if episode_length >= max_episode_length:
                    break
                else:
                    continue

            if episode_length >= max_episode_length:
                done = True

            eef = next_obs.gripper_pose[:3]
            next_obs = np.concatenate((eef, target))
            
            agent.step(obs, action[:3], reward, next_obs, done)
            obs = next_obs
            score += reward

        experiment.log_metric('score', score)
        experiment.log_metric('episode_length', episode_length)

        scores.append(score)
        mean_score = np.mean(scores)
        print(f'\rEpisode: {i}\tAverage Score: {mean_score:.2f}', end="")
        if i % 10 == 0:
            print(f'\rEpisode: {i}\tAverage Score: {mean_score:.2f}')
        
