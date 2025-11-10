import os 
import numpy as np
import pandas as pd
import gymnasium as gym

from mani_skill.utils import gym_utils
from mani_skill.envs.sapien_env import BaseEnv

from wrapper_v1 import MyRecordEpisode
from env_widowx_v8 import PickCubeEnv


def collect_trajectories(verbose=True, seed=0, num_trajs=5, strat_traj=0):

    env_id = 'PickCubeWidowX-v8'
    env_kwargs = {
        'obs_mode': 'state', 
        'reward_mode': None, 
        'control_mode': None,
        'control_mode': 'pd_ee_target_delta_pose', 
        'render_mode': 'rgb_array', 
        'human_render_camera_configs': dict(width=1024, height=1024, shader_pack='rt'), 
        'num_envs': 1, 
        'sim_backend': 'auto', 
        'render_backend': 'gpu', 
        'enable_shadow': True, 
        'parallel_in_single_scene': False, 
        'robot_uids': 'mywidowx250s', 
        'max_episode_steps': 200,
        'split': 'train',
    }

    env: BaseEnv = gym.make(
        env_id,
        **env_kwargs,
    )

    count_traj = strat_traj
    record_dir = "dataset_sim_touch/{robot_uid}_v8"
    if record_dir:
        record_dir = record_dir.format(robot_uid=env_kwargs["robot_uids"])
        env = MyRecordEpisode(
            env, 
            record_dir, 
            info_on_video=True, 
            save_trajectory=True, 
            max_steps_per_video=gym_utils.find_max_episode_steps_value(env),
        )
        env._episode_id = count_traj - 1
        env._video_id = count_traj - 1

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, info = env.reset(seed=seed, options=dict(reconfigure=True))
    # pre-defined steps of operation
    counts = np.array([50, 5, 20, 50])
    stages = np.cumsum(counts)
    count = 0

    records = []

    while True:

        stage = (count > stages).sum()

        if stage == 0:
            # fetch cube
            eef_pos = env.unwrapped.agent.tcp_pos
            cube_pos = env.unwrapped.cube.pose.p
            # eef_pos, cube_pos: [1, 3]
            rel_pos = (cube_pos - eef_pos) / counts[stage] * 30

            action = np.zeros(7)
            action[:3] = rel_pos
            action[-1] = 1.0
            # avoid crushing table
            if eef_pos[0, 2] < 0.03:
                action[2] = 0.0 
        elif stage == 1:
            # pick cube
            action = np.zeros(7)
            action[-1] = 0.0

        elif stage == 2:
            # lift the cube
            eef_pos = env.unwrapped.agent.tcp_pos
            target_z = 0.06
            rel_z = (target_z - eef_pos[0, 2]) / counts[stage] * 30

            action = np.zeros(7)
            action[2] = rel_z
            action[-1] = 0.0
        elif stage == 3:
            # move to target card
            eef_pos = env.unwrapped.agent.tcp_pos
            card_pos = env.unwrapped.target_card.pose.p
            # eef_pos, card_pos: [1, 3]
            card_pos[:, 2] += 0.05
            rel_pos = (card_pos - eef_pos) / counts[stage] * 30

            action = np.zeros(7)
            action[:3] = rel_pos
            action[-1] = 0.0
        elif stage == 4:
            # release cube
            action = np.zeros(7)
            action[-1] = 1.0

        count += 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if verbose:
            print("obs", obs)
            print("action", action)
            print("reward", reward)
            print("terminated", terminated)
            print("truncated", truncated)
            print("info", info)

        if (terminated | truncated).any():

            records.append(dict(
                num_steps = info['elapsed_steps'].item(),
                task=info['task'],
                success=info['success'].item(),
                is_obj_placed=info['is_obj_placed'].item(),
                is_robot_static=info['is_robot_static'].item(),
                is_grasped=info['is_grasped'].item(),
                card_0=env.unwrapped.selected_meta[0]['desc'],
                card_1=env.unwrapped.selected_meta[1]['desc'],
                card_2=env.unwrapped.selected_meta[2]['desc'],
                target=env.unwrapped.target_id,
                cube_position=env.unwrapped.cube.pose.p.numpy().flatten().tolist(),
            ))
            pd.DataFrame(records).to_csv(os.path.join(record_dir, f'generation_records.csv'))

            count_traj += 1
            if count_traj < num_trajs:
                obs, info = env.reset(seed=seed + count_traj, options=dict(reconfigure=True))
                # pre-defined steps of operation
                counts = np.array([50, 5, 20, 50])
                stages = np.cumsum(counts)
                count = 0
            else:
                break

    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


def collect():

    collect_trajectories(num_trajs=500, strat_traj=0)


if __name__ == "__main__":

    collect()