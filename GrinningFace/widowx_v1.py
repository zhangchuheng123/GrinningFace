import torch
import sapien
import numpy as np
from copy import deepcopy

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PDEEPoseControllerConfig, PDJointPosMimicControllerConfig, deepcopy_dict
from mani_skill.agents.controllers.base_controller import CombinedController
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


# TODO (stao) (xuanlin): model it properly based on real2sim
@register_agent(asset_download_ids=["widowx250s"])
class MyWidowX250S(BaseAgent):
    uid = "mywidowx250s"
    urdf_path = f"{ASSET_DIR}/robots/widowx/wx250s.urdf"
    urdf_config = dict()

    arm_joint_names = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]
    gripper_joint_names = ["left_finger", "right_finger"]

    # chuheng
    ee_link_name = "ee_gripper_link"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100
    # end chuheng

    def reset(self, init_qpos: torch.Tensor = None):
        """
        Reset the robot to a clean state with zero velocity and forces.

        Args:
            init_qpos (torch.Tensor): The initial qpos to set the robot to. If None, the robot's qpos is not changed.
        """

        if init_qpos is not None:
            self.robot.set_qpos(init_qpos)
        else:
            qpos = torch.zeros(self.robot.max_dof, device=self.device)
            # girpper face downwards
            # qpos[1] = np.pi / 4
            # qpos[2] = - np.pi / 4
            qpos[4] = np.pi / 180 * 90
            # open gripper
            qpos[-2] = 0.037
            qpos[-1] = 0.037
            self.robot.set_qpos(qpos)

        self.robot.set_qvel(torch.zeros(self.robot.max_dof, device=self.device))
        self.robot.set_qf(torch.zeros(self.robot.max_dof, device=self.device))

    def set_control_mode(self, control_mode: str = None):
        """Sets the controller to an pre-existing controller of this agent.
        This does not reset the controller. If given control mode is None, will set to the default control mode."""

        if control_mode is None:
            control_mode = self._default_control_mode
        assert (
            control_mode in self.supported_control_modes
        ), "{} not in supported modes: {}".format(
            control_mode, self.supported_control_modes
        )
        self._control_mode = control_mode
        # create controller on the fly here
        if control_mode not in self.controllers:
            config = self._controller_configs[self._control_mode]
            balance_passive_force = True
            if isinstance(config, dict):
                if "balance_passive_force" in config:
                    balance_passive_force = config.pop("balance_passive_force")

                self.controllers[control_mode] = CombinedController(
                    config,
                    self.robot,
                    self._control_freq,
                    scene=self.scene,
                )
            else:
                self.controllers[control_mode] = config.controller_cls(
                    config, self.robot, self._control_freq, scene=self.scene
                )

            self.controllers[control_mode].set_drive_property()
            if balance_passive_force:
                # NOTE (stao): Balancing passive force is currently not supported in PhysX, so we work around by disabling gravity
                if not self.scene._gpu_sim_initialized:
                    for link in self.robot.links:
                        link.disable_gravity = True
                else:
                    for link in self.robot.links:
                        if link.disable_gravity.all() != True:
                            logger.warning(
                                f"Attemped to set control mode and disable gravity for the links of {self.robot}. However the GPU sim has already initialized with the links having gravity enabled so this will not work."
                            )

    @property
    def _controller_configs(self):

        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"left_finger": {"joint": "right_finger"}},
        )

        controller_configs = dict(
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        self.finger1_link = self.robot.links_map["left_finger_link"]
        self.finger2_link = self.robot.links_map["right_finger_link"]

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def _after_init(self):

        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_static(self, threshold: float = 0.2):

        # the meaning of each vel is not checked 
        # get_qvel(): [1, 8]
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose