from comet_ml import Experiment
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
import argparse
from collections import deque
from pyquaternion import Quaternion
from ddpg_her_agent import DDPG_HERAgent


def get_experiment(exp_name, exp_tags):
    experiment = Experiment(
        api_key="4iMhvGHg82QsAVYdkRfxTzesW",
        project_name="rl",
        workspace="anilz",
        auto_output_logging="simple",
        parse_args=False,
        auto_metric_logging=False
    )

    experiment.set_name(exp_name)
    experiment.add_tags(exp_tags)

    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_comet", default=False, action='store_true')
    parser.add_argument('--lr_critic', default=1e-3, type=float, help='Q net learning rate')
    parser.add_argument('--lr_actor', default=1e-4, type=float, help='policy net learning rate')
    parser.add_argument('--buffer_size', default=500000, type=int, help='memory size')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--polyak', default=0.995, type=float, help='moving average for target network')
    parser.add_argument("--num_episodes", type=int, default=100000)
    parser.add_argument("--max_episode_length", type=int, default=2000)
    parser.add_argument("--action_noise", type=float, default=0.02)
    parser.add_argument('--start_steps', default=100000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument("--update_every", type=float, default=10)
    parser.add_argument("--update_after", type=float, default=1000)
    args = parser.parse_args()

    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(True)

    action_mode = ActionMode(ArmActionMode.EE_POSE_EE_FRAME)
    env = Environment(action_mode, obs_config=obs_config, headless=False)
    env.launch()

    task = env.get_task(ReachTarget)
    experiment = get_experiment("RLBench Reacher", ["DDPG", "HER", "EE_POSE"]) if args.use_comet else None
    agent = DDPG_HERAgent(obs_dim=6, act_dim=3, act_limits=0.02, buffer_size=args.buffer_size, batch_size=args.batch_size,
                        gamma=args.discount, polyak=args.polyak, lr_actor=args.lr_actor, lr_critic=args.lr_critic, 
                        start_steps=args.start_steps, update_every=args.update_every, update_after=args.update_after,
                        experiment=experiment)

    if args.use_comet:
        experiment.log_parameters({
                "num_episodes": args.num_episodes,
                "max_episode_length": args.max_episode_length,
                "action_noise": args.action_noise,
                "buffer_size": args.buffer_size,
                "batch_size": args.batch_size,
                "discount": args.discount,
                "polyak": args.polyak,
                "lr_critic": args.lr_critic,
                "lr_actor": args.lr_actor,
                "start_steps": args.start_steps,
                "update_every": args.update_every,
                "update_after": args.update_after,
        })

    else:
        agent.load_parameters()

    scores = deque(maxlen=10)
    for i in range(1, args.num_episodes+1):
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

            action = agent.act(obs, noise=args.action_noise)
            action = np.concatenate((action, eef_ori, [gripper_state]))
            try:
                next_obs, reward, done = task.step(action)
            except:
                if episode_length >= args.max_episode_length:
                    break
                else:
                    continue

            if episode_length >= args.max_episode_length:
                done = True

            eef = next_obs.gripper_pose[:3]
            next_obs = np.concatenate((eef, target))
            
            if args.use_comet:
                agent.step(obs, action[:3], reward, next_obs, done)
            obs = next_obs
            score += reward

        if args.use_comet:
            experiment.log_metric('score', score)
            experiment.log_metric('episode_length', episode_length)

        scores.append(score)
        mean_score = np.mean(scores)
        print(f'\rEpisode: {i}\tAverage Score: {mean_score:.2f}', end="")
        if i % 10 == 0:
            print(f'\rEpisode: {i}\tAverage Score: {mean_score:.2f}')

        if args.use_comet and mean_score == 1 and i > 100:
            agent.save_parameters()
