import os
import sys
import math
import time
import torch
import hydra
import pickle
import inspect 
import logging
import numpy as np
import pandas as pd
from PIL import Image
import gymnasium as gym
from collections import deque
from datetime import datetime
from einops import rearrange, repeat
# from matplotlib import pyplot as plt
from torchvision.transforms import v2
from mani_skill.utils import gym_utils
from omegaconf import OmegaConf, open_dict

from wrapper_v1 import MyRecordEpisode
from env_widowx_v8 import PickCubeEnv


# myVLA (oxe) + full finetune 
MODEL_STR = "MODEL_TO_TEST"
CONFIG_NAME = "config_used_for_model_inference.yaml"
os.environ['INFERENCE_CHECKPOINT'] = "/path/to/checkpoint.pt"

RUN_ATTENTION = False
ATTN_METHOD = 'word_attn'


# for logging
logger = logging.getLogger('uvicorn.info')
log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)


class SimulatorEvaluator:
    def __init__(self, cfg: OmegaConf):
        """
        A simple server for models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """

        with open_dict(cfg):
            time_str = datetime.now().strftime("%Y%m%d%H%M%S")
            cfg.gpu_id = 0
            cfg.multi_gpu = False
            cfg.log_dir = f'/mnt/chuheng_data/exp_pi/simulation/{MODEL_STR}/touch_sim_{time_str}'

        self.cfg = cfg
        # Import your own agent here
        self.agent = InferenceAgent(cfg)

        self.counter = 0
        self.img_transform = v2.Resize((224, 224))


    def predict_action(self, image, proprio, task_description, run_attention=RUN_ATTENTION):
        """
        image: torch uint8 [1, H, W=H, 3] unknown device 
        proprio: torch float32 [1, 7] unnormalized unknown device
        task_description: str
        """

        # debug
        if self.counter == 0:
            Image.fromarray(image.squeeze(0).cpu().numpy()).save('verify_primary.png')
            if run_attention:
                # not implemented
                self.run_attention()

        image = rearrange(image.cpu(), 'B H W C -> B C H W')
        image = self.img_transform(image)

        inputs = {
            'image': image,
            'image_wrist': None,
            'proprio': rearrange(proprio.cpu(), '(B T) d -> B T d', B=1, T=1),
            'task_description': [task_description],
        }

        action, info = self.agent.inference(**inputs)
        action = action[0, :]

        return action 
    
    def run(self, split='val', num_trajs=5, seed=10, verbose=True):

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
            'split': split,
        }

        env = gym.make(
            env_id,
            **env_kwargs,
        )

        env = MyRecordEpisode(
            env, 
            os.path.join(self.cfg.log_dir, split), 
            info_on_video=True, 
            save_trajectory=False, 
            save_video=False,
            max_steps_per_video=gym_utils.find_max_episode_steps_value(env),
        )

        records = []

        count_traj = 0

        obs, info = env.reset(seed=seed, options=dict(reconfigure=True))
        task = info['task']
        proprio = info['proprio']
        image = env.render()
        self.counter = 0
        self.timer = deque(maxlen=100)
        tic = time.time()

        while True:
            action = self.predict_action(image, proprio, task)
            obs, reward, terminated, truncated, info = env.step(action)
            proprio = info['proprio']
            image = env.render()
            self.counter += 1

            if verbose:
                print("step", count_traj, self.counter)
                print("action", action)

            if (terminated | truncated).any():

                records.append(dict(
                    model=MODEL_STR,
                    split=split,
                    num_steps = info['elapsed_steps'].item(),
                    task=task,
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
                pd.DataFrame(records).to_csv(os.path.join(self.cfg.log_dir, f'records_{split}.csv'))

                count_traj += 1
                if count_traj < num_trajs:
                    obs, info = env.reset(seed=seed + count_traj, options=dict(reconfigure=True))
                    task = info['task']
                    proprio = info['proprio']
                    image = env.render()
                    self.counter = 0
                else:
                    break

            self.timer.append(time.time() - tic)
            tic = time.time()
            print(f"[SimulatorEvaluator] Average step time: {np.mean(self.timer):.3f}s over last {len(self.timer)} steps", end='\r')

        env.close()
        return records
    
@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config", "train"),
    config_name=CONFIG_NAME,
)
def main(cfg: OmegaConf) -> None:
    # 在正式运行前等待 GPU 显存占用低于 50%
    def wait_for_gpu_memory(threshold: float = 0.5, interval: int = 30, device: int = 0):
        """轮询 GPU 显存占用，低于阈值后继续。
        threshold: 占用比例上限 (0~1)
        interval: 轮询间隔（秒）
        device: GPU 编号
        """
        import time
        if not torch.cuda.is_available():
            print("[GPU 内存检查] CUDA 不可用，跳过检查。")
            return
        dev = torch.device(f'cuda:{device}')
        while True:
            try:
                free, total = torch.cuda.mem_get_info(dev)
                used = total - free
                usage = used / total
                if usage < threshold:
                    print(f"[GPU 内存检查] 当前占用 {usage*100:.1f}% < {threshold*100:.0f}%，继续运行。")
                    break
                else:
                    print(f"[GPU 内存检查] 当前占用 {usage*100:.1f}% >= {threshold*100:.0f}%，等待 {interval}s 后重试。")
            except Exception as e:
                print(f"[GPU 内存检查] 获取信息失败: {e}，直接继续。")
                break
            time.sleep(interval)

    wait_for_gpu_memory(threshold=0.5, interval=30, device=0)

    eval = SimulatorEvaluator(cfg)
    records = []
    records.extend(eval.run(split='val', num_trajs=100))
    records.extend(eval.run(split='train', num_trajs=100))
    records.extend(eval.run(split='id', num_trajs=100))


if __name__ == "__main__":
    main()
