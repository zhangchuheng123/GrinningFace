import json
import torch
import sapien
import random
import pickle
import sapien
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import Any, Dict, Union, Optional


from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents import REGISTERED_AGENTS
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.agents.robots import SO100, Fetch, Panda, XArm6Robotiq
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS

from widowx_v1 import MyWidowX250S


PICK_CUBE_DOC_STRING = \
"""**Task Description:**
A simple task where the objective is to grasp a blue cube with the robot arm and move it to a instructed card=. 

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table (TBD)
- the cube's z-axis rotation is randomized to a random angle (TBD)
- the images on the card and the target card is randomly chosen

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal card
- the robot is static (q velocity < 0.2)

v1:
- initial environment with bugs fixed
- used for collecting panda_v2 and panda_v3
v2:
- match bridge visually (not completed)
v3:
- randomized cube and card locations
- used for collecting panda_v4
v4: 
- implement widowx
widowx_v5:
- implement mywidowx250s
widowx_v6:
- implement mywidowx250s with parallel envs
widowx_v7:
- implement mywidowx250s with parallel envs
- 1024x1024
widowx_v8:
- implement mywidowx250s
- PaliGemma relabeled and filtered meta
widowx_v9:
- implement mywidowx250s
- PaliGemma relabeled and filtered meta
- more discretized actions
"""

XO = 0.2
YO = 0.7
DEMO = False


@register_env("PickCubeWidowX-v8", max_episode_steps=200)
class PickCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = [
        "panda",                    # has pd_ee_target_delta_pos
        "fetch",                    # has pd_ee_target_delta_pos (an mobile robot)
        "xarm6_robotiq",            # has pd_ee_target_delta_pos
        "so100",                    # only pd_joint_delta_pos, pd_joint_pos, pd_joint_target_delta_pos
        "mywidowx250s",             # only pd_joint_pos, pd_joint_delta_pos
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100, MyWidowX250S]

    robot_root_xyz = [-0.45 + XO, 0 + YO, 0.1]

    cube_half_size = 0.015
    cube_spawn_center = (-0.18 + XO, 0 + YO)
    cube_spawn_half_size = 0.05

    card_x_coord = 0.0 + XO
    card_y_coords = [0.2 + YO, 0 + YO, -0.2 + YO]
    card_spawn_half_size = 0.02
    card_half_size = 0.07
    card_thick = 0.001

    camera_fov = 1.2

    if DEMO:
        # for global view point
        camera_from = [XO - 0.6, YO - 0.25, 0.7]
        camera_to   = [XO - 0.05, YO + 0.05, 0.0]
    else:
        # Matches the bridge view point as much as possible
        camera_from = [XO - 0.3 - 0.05, YO - 0.16 - 0.05, 0.564]
        camera_to   = [XO - 0.05, YO - 0.05, 0.0]


    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, split='train', **kwargs):

        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        # human camera is used for .render()
        # Matches the bridge view point as much as possible
        self.human_cam_eye_pos = self.camera_from   
        self.human_cam_target_pos = self.camera_to
        self.split = split

        if split == 'id':
            asset_path = Path('/mnt/chuheng_data/dataset_sim_touch/mywidowx250s_v8/metadata_train_sposeulerg_0_1_aposeulerg_4_v3.pkl')
            with open(asset_path, 'rb') as f:
                meta = pickle.load(f)
            traj_ids = sorted([int(p['image_path'].split('/')[-2]) for p in meta])

            generation_record_path = Path('/mnt/chuheng_data/dataset_sim_touch/mywidowx250s_v8/generation_records.csv')
            generation_record = pd.read_csv(generation_record_path, index_col=0)
            generation_record = generation_record.loc[traj_ids]
            self.id_record = generation_record.to_dict('records')

        self.asset_path = Path('/mnt/chuheng_data/dataset_emoji')
        if split == 'train':
            self.meta_path = self.asset_path / 'meta_relabel_dedup_train.json'
        elif split == 'val':
            self.meta_path = self.asset_path / 'meta_relabel_dedup_val.json'
        elif split in ['all', 'id']:
            self.meta_path = self.asset_path / 'meta_relabel_dedup_all.json'

        if not self.meta_path.exists():
            self.download_emoji()

        with open(self.meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

        if split == 'id':
            self.meta = {item['desc']: item for item in self.meta}

        self.target_id = None
        self.selected_meta = []
        self.cards = []
        self.target_card = None
        self.task_description = ""

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 224, 224, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 1024, 1024, self.camera_fov, 0.01, 100)

    def _load_agent(self, options: dict, build_separate: bool = False):
        """
        loads the agent/controllable articulations into the environment. The default function provides a convenient way to setup the agent/robot by a robot_uid
        (stored in self.robot_uids) without requiring the user to have to write the robot building and controller code themselves. For more
        advanced use-cases you can override this function to have more control over the agent/robot building process.

        Args:
            options (dict): The options for the environment.
            initial_agent_poses (Optional[Union[sapien.Pose, Pose]]): The initial poses of the agent/robot. Providing these poses and ensuring they are picked such that
                they do not collide with objects if spawned there is highly recommended to ensure more stable simulation (the agent pose can be changed later during episode initialization).
            build_separate (bool): Whether to build the agent/robot separately. If True, the agent/robot will be built separately for each parallel environment and then merged
                together to be accessible under one view/object. This is useful for randomizing physical and visual properties of the agent/robot which is only permitted for
                articulations built separately in each environment.
        """
        
        # Environment related
        initial_agent_poses = sapien.Pose(p=[-0.615, 0, 0])

        # Super class method
        agents = []
        robot_uids = self.robot_uids
        if not isinstance(initial_agent_poses, list):
            initial_agent_poses = [initial_agent_poses]
        if robot_uids == "none" or robot_uids == ("none", ):
            self.agent = None
            return

        if robot_uids is not None:
            if not isinstance(robot_uids, tuple):
                robot_uids = [robot_uids]
            for i, robot_uid in enumerate(robot_uids):
                if isinstance(robot_uid, type(BaseAgent)):
                    agent_cls = robot_uid
                else:
                    if robot_uid == 'mywidowx250s':

                        initial_agent_pose = sapien.Pose(p=self.robot_root_xyz)
                        agent: BaseAgent = MyWidowX250S(
                            self.scene,
                            self._control_freq,         # 20
                            self._control_mode,         # pd_ee_target_delta_pose
                            agent_idx=i if len(robot_uids) > 1 else None,
                            initial_pose=initial_agent_pose,
                            build_separate=build_separate,
                        )
                    elif robot_uid in REGISTERED_AGENTS:
                        agent_cls = REGISTERED_AGENTS[robot_uid].agent_cls
                        agent: BaseAgent = agent_cls(
                            self.scene,
                            self._control_freq,
                            self._control_mode,
                            agent_idx=i if len(robot_uids) > 1 else None,
                            initial_pose=initial_agent_poses[i] if initial_agent_poses is not None else None,
                            build_separate=build_separate,
                        )
                    else:
                        raise NotImplementedError(f"{robot_uid} is not implemented")
                    agents.append(agent)
        if len(agents) == 1:
            self.agent = agents[0]
        else:
            self.agent = MultiAgent(agents)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0.1, 0.1, 0.5, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        # current_file_path = Path(__file__)
        # wooden_file = current_file_path.parent / 'texture/simple-wood-texture.jpg'
        # wooden_texture = sapien.render.RenderTexture2D(str(wooden_file))
        wooden_mat = sapien.render.RenderMaterial(base_color=[214/256, 183/256, 146/256, 1])
        # wooden_mat.set_base_color_texture(wooden_texture)

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.5, 1.3, 2.0])
        builder.add_box_visual(half_size=[0.5, 1.3, 2.0], material=wooden_mat)
        builder.set_initial_pose(sapien.Pose(p=[0.5 + 0.5, 0 - 0.05, 2.0 - 0.91964257]))
        self.background = builder.build(name='background')

        # wall_file = current_file_path.parent / 'texture/simple-wall-texture.jpg'
        # wall_texture = sapien.render.RenderTexture2D(str(wall_file))
        wall_mat = sapien.render.RenderMaterial(base_color=[98/256, 117/256, 188/256, 1])
        # wall_mat.set_base_color_texture(wall_texture)

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[1.3, 0.5, 2.0])
        builder.add_box_visual(half_size=[1.3, 0.5, 2.0], material=wall_mat)
        builder.set_initial_pose(sapien.Pose(p=[0, 1.3 + 0.5 - 0.05, 2.0 - 0.91964257]))
        self.wall = builder.build(name='wall')

        self.cards = []
        self.selected_meta = []

        if self.split == 'id':
            combination = random.choice(self.id_record)
            self.target_id = combination['target']
            meta_items = [self.meta[combination['card_0']], self.meta[combination['card_1']], self.meta[combination['card_2']]]
        else:
            self.target_id = random.choice([0, 1, 2])
            meta_items = random.sample(self.meta, 3)

        if DEMO:
            # ambulance, rocket, carousel
            meta_items = [self.meta[81], self.meta[91], self.meta[78]]

        for i, y_axis in enumerate(self.card_y_coords):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[self.card_half_size, self.card_half_size, self.card_thick])

            meta_item = meta_items[i]
            self.selected_meta.append(meta_item)
            input_path = meta_item['full_path']

            texture_img = sapien.render.RenderTexture2D(input_path)
            mat = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1])
            mat.set_base_color_texture(texture_img)
            builder.add_box_visual(half_size=[self.card_half_size, self.card_half_size, self.card_thick], material=mat)
            builder.set_initial_pose(sapien.Pose(p=[self.card_x_coord, y_axis, self.card_thick]))
            card = builder.build(name=f'card_{i}')
            self.cards.append(card)

            if self.target_id == i:
                self.task_description = f"pick the block and place it on {meta_item['paligemma_label']}"
                self.target_card = card

    @staticmethod
    def download_emoji_image(emoji, style='apple'):

        url = f"https://emojicdn.elk.sh/{emoji['char']}?style={style}"
        response = requests.get(url)

        try:
            response = requests.get(url)
            if response.status_code == 200:
                return BytesIO(response.content)
        except Exception:
            print(f"Failed to download emoji {emoji['code']}")
        return None
    
    def download_emoji(self):

        current_file_path = Path(__file__)
        emoji_list_file = current_file_path.parent / 'emoji/emoji_list.json'

        self.asset_path.mkdir(parents=True, exist_ok=True)

        records = []

        with open(emoji_list_file, 'r', encoding='utf-8') as f:
            emoji_data = json.load(f)

        meta_path = self.asset_path / "meta.json"
        meta_train_path = self.asset_path / "meta_train.json"
        meta_val_path = self.asset_path / "meta_val.json"

        for emoji in tqdm(emoji_data, desc='downloading emoji'):

            for style in ['apple', 'google', 'facebook', 'twitter']:
                img = self.download_emoji_image(emoji, style)
                try:
                    img = Image.open(img)
                    if img.size != (160, 160):
                        img.resize((160, 160))
                    break
                except:
                    print(f"Failed on {emoji['paligemma_label']} with style {style}")
                    continue

            if not isinstance(img, Image.Image):
                print(f"Failed on {emoji['paligemma_label']} with all styles")
                continue
        
            img = img.convert('RGBA')
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            white_bg = Image.new('RGB', img.size, (255, 255, 255))
            white_bg.paste(img, mask=img.split()[3])

            enlarge = Image.new('RGB', (660, 560), (255, 255, 255))
            enlarge.paste(white_bg.rotate(270), (170, 200))

            filename = emoji['desc'] + '.png'
            output_path = self.asset_path / filename
            enlarge.save(output_path, 'PNG')
            emoji['filename'] = filename
            emoji['full_path'] = str(output_path)
            records.append(emoji)

        with open(meta_path, 'w') as f:
            json.dump(records, f, indent=4)

        records_train = [item for item in records if (item['no'] >= 562) and (item['no'] < 662)]
        with open(meta_train_path, 'w') as f:
            json.dump(records_train, f, indent=4)

        records_val = [item for item in records if not ((item['no'] >= 562) and (item['no'] < 662))]
        with open(meta_val_path, 'w') as f:
            json.dump(records_val, f, indent=4)
            
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size

            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            q0 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=True)
            y_axes = self.card_y_coords
            for i, card in enumerate(self.cards):
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = (
                    torch.rand((b, 2)) * self.card_spawn_half_size * 2
                    - self.card_spawn_half_size
                )
                xyz[:, 0] += self.card_x_coord
                xyz[:, 1] += y_axes[i]
                xyz[:, 2] = self.card_thick

                card.set_pose(Pose.create_from_pq(xyz, q0))

    @property
    def proprio_pos_euler_gripper(self):

        if self.robot_uids == 'panda':
            gripper_width = 0.08
        elif self.robot_uids == 'mywidowx250s':
            # TODO: Start to take effect from next version
            # gripper_width = 0.074
            gripper_width = 0.08

        proprio = self.agent.tcp_pose.raw_pose.clone()
        euler = Rotation.from_quat(self.agent.tcp_pose.q).as_euler('xyz', degrees=False)
        proprio[:, 3:-1] = torch.tensor(euler).to(proprio)
        proprio[:, -1] = np.linalg.norm(self.agent.finger1_link.pose.p - self.agent.finger2_link.pose.p) / gripper_width

        return proprio

    def _get_obs_extra(self, info: Dict):
        
        obs = dict(
            proprio=self.proprio_pos_euler_gripper,
        )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.target_card.pose.p - self.cube.pose.p, axis=1)
            <= self.card_half_size * np.sqrt(2)
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)

        return {
            "proprio": self.proprio_pos_euler_gripper,
            "task": self.task_description,
            "success": is_obj_placed & is_robot_static & (not is_grasped),
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        # Not Implemented
        raise NotImplementedError()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return torch.ones((action.shape[0], )) if info['success'] else torch.zeros((action.shape[0], ))