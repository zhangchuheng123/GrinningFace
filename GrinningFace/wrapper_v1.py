import numpy as np
from PIL import Image

from mani_skill.utils import common
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.visualization.misc import put_info_on_image, tile_images


class MyRecordEpisode(RecordEpisode):

    def _save_single_image(self, img, path):

        Image.fromarray(img).save(path)

    def step(self, action):

        image_output_path = self.output_dir / str(self._episode_id + 1) / 'images' / 'primary_image'
        if self.save_video and self._video_steps == 0:
            # save the first frame of the video here (s_0) instead of inside reset as user
            # may call env.reset(...) multiple times but we want to ignore empty trajectories
            image_output_path.mkdir(parents=True, exist_ok=True)
            img, oimg = self.capture_image()
            if self.save_trajectory:
                self._save_single_image(oimg, image_output_path / f"{self._video_steps}.png")
            self.render_images.append(img)

        obs, rew, terminated, truncated, info = super(RecordEpisode, self).step(action)
        done = terminated | truncated

        if self.save_trajectory:
            state_dict = self.base_env.get_state_dict()
            if self.record_env_state:
                self._trajectory_buffer.state = common.append_dict_array(
                    self._trajectory_buffer.state,
                    common.to_numpy(common.batch(state_dict)),
                )
            self._trajectory_buffer.observation = common.append_dict_array(
                self._trajectory_buffer.observation,
                common.to_numpy(common.batch(obs)),
            )

            self._trajectory_buffer.action = common.append_dict_array(
                self._trajectory_buffer.action,
                common.to_numpy(common.batch(action)),
            )
            if self.record_reward:
                self._trajectory_buffer.reward = common.append_dict_array(
                    self._trajectory_buffer.reward,
                    common.to_numpy(common.batch(rew)),
                )
            self._trajectory_buffer.terminated = common.append_dict_array(
                self._trajectory_buffer.terminated,
                common.to_numpy(common.batch(terminated)),
            )
            self._trajectory_buffer.truncated = common.append_dict_array(
                self._trajectory_buffer.truncated,
                common.to_numpy(common.batch(truncated)),
            )
            self._trajectory_buffer.done = common.append_dict_array(
                self._trajectory_buffer.done,
                common.to_numpy(common.batch(done)),
            )
            if "success" in info:
                self._trajectory_buffer.success = common.append_dict_array(
                    self._trajectory_buffer.success,
                    common.to_numpy(common.batch(info["success"])),
                )
            else:
                self._trajectory_buffer.success = None
            if "fail" in info:
                self._trajectory_buffer.fail = common.append_dict_array(
                    self._trajectory_buffer.fail,
                    common.to_numpy(common.batch(info["fail"])),
                )
            else:
                self._trajectory_buffer.fail = None

        if self.save_video:
            self._video_steps += 1
            if self.info_on_video:
                image, oimg = self.capture_image(info)
            else:
                image, oimg = self.capture_image()

            if (not done) and self.save_trajectory:
                self._save_single_image(oimg, image_output_path / f"{self._video_steps}.png")
            self.render_images.append(image)
            if (
                self.max_steps_per_video is not None
                and self._video_steps >= self.max_steps_per_video
            ):
                self.flush_video()

        self._elapsed_record_steps += 1
        return obs, rew, terminated, truncated, info

    def flush_trajectory(self, **kwargs):

        traj_path = self.output_dir / str(self._episode_id + 1)

        # Remove the last frame since no action is conduced from that state
        # Only the last 7 dimensions are proprio state
        proprios = self._trajectory_buffer.observation[:-1, 0, -7:]
        np.save(traj_path / 'left_arm_poseuler_arm.npy', proprios)

        # Remove the first action since it is a random action
        actions = self._trajectory_buffer.action[1:, 0]
        np.save(traj_path / 'action.npy', actions)

        with open(traj_path / 'task_instruction.txt', 'w') as f:
            f.write(self.unwrapped.task_description)

        super().flush_trajectory(**kwargs)

    def capture_image(self, info=None):
        img = self.env.render()
        img = common.to_numpy(img)

        if len(img.shape) == 3:
            img = img[None]

        oimg = img[0].copy()

        if info is not None:
            texts = []
            if "elapsed_steps" in info:
                texts.append(f"elapsed_steps: {info['elapsed_steps'].item()}")
            if "task" in info:
                texts.append(f"task: {info['task']}")
            if "proprio" in info:
                list_str = ["{0:0.4f}".format(num) for num in info["proprio"].numpy().flatten().tolist()]
                list_str = ', '.join(list_str)
                texts.append(f"proprio: [{list_str}]")
            
            if not isinstance(info['is_grasped'], float):
                texts.append((
                    f"grasped: {info['is_grasped'].item()} "
                    f"placed: {info['is_obj_placed'].item()} "
                    f"static: {info['is_robot_static'].item()}"
                ))
                texts.append(f"success: {info['success'].item()}")

            for i in range(len(img)):
                info_item = {}
                img[i] = put_info_on_image(img[i], info_item, extras=texts)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = tile_images(img, nrows=self.video_nrows)

        return img, oimg