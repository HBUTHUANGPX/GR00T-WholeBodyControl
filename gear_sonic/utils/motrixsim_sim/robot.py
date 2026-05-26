"""MotrixSim G1 state and actuator helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


BODY_JOINT_KEYWORDS: tuple[str, ...] = (
    "hip",
    "knee",
    "ankle",
    "waist",
    "shoulder",
    "elbow",
    "wrist",
)


class MotrixG1Robot:
    """Access G1 joint, link, and actuator state in MotrixSim.

    Responsibilities:
        Preserve the joint ordering used by the existing MuJoCo simulator while
        hiding MotrixSim-specific API calls from the environment loop.
    Preconditions:
        ``model`` was loaded from the current Gear Sonic G1 MJCF scene.
    Postconditions:
        Callers can read body/hand states and set actuator controls in the
        MotrixSim model's actuator order.
    """

    def __init__(self, model, config: dict) -> None:
        """Create a MotrixSim G1 accessor.

        Preconditions:
            The model has ``pelvis`` and ``torso_link`` links and 43 actuators.
        Postconditions:
            Joint name groups and actuator order are validated.
        """

        self.model = model
        self.config = config
        self.num_body_joints = int(config["NUM_JOINTS"])
        self.num_hand_joints = int(config.get("NUM_HAND_JOINTS", 0))
        self.num_hand_motors = int(config.get("NUM_HAND_MOTORS", 0))
        self.default_body_angles = np.asarray(config["DEFAULT_DOF_ANGLES"], dtype=np.float32)
        self.body = model.get_body("pelvis")

        self.body_joint_names = [
            name
            for name in model.joint_names
            if any(keyword in name for keyword in BODY_JOINT_KEYWORDS)
        ]
        self.left_hand_joint_names = [name for name in model.joint_names if name.startswith("left_hand")]
        self.right_hand_joint_names = [name for name in model.joint_names if name.startswith("right_hand")]
        if len(self.body_joint_names) != self.num_body_joints:
            raise ValueError(
                f"Expected {self.num_body_joints} body joints, found {len(self.body_joint_names)}"
            )
        if len(self.left_hand_joint_names) != self.num_hand_joints:
            raise ValueError(
                f"Expected {self.num_hand_joints} left hand joints, found {len(self.left_hand_joint_names)}"
            )
        if len(self.right_hand_joint_names) != self.num_hand_joints:
            raise ValueError(
                f"Expected {self.num_hand_joints} right hand joints, found {len(self.right_hand_joint_names)}"
            )

        self.body_actuator_names = [self._actuator_name_from_joint(name) for name in self.body_joint_names]
        self.left_hand_actuator_names = [
            self._actuator_name_from_joint(name) for name in self.left_hand_joint_names
        ]
        self.right_hand_actuator_names = [
            self._actuator_name_from_joint(name) for name in self.right_hand_joint_names
        ]
        self._all_actuator_names = list(model.actuator_names)
        self._joint_name_to_dof_index = {name: index for index, name in enumerate(model.joint_names)}
        self._body_joint_dof_indices = np.asarray(
            [self._joint_name_to_dof_index[name] for name in self.body_joint_names], dtype=np.int64
        )
        self._left_hand_joint_dof_indices = np.asarray(
            [self._joint_name_to_dof_index[name] for name in self.left_hand_joint_names], dtype=np.int64
        )
        self._right_hand_joint_dof_indices = np.asarray(
            [self._joint_name_to_dof_index[name] for name in self.right_hand_joint_names], dtype=np.int64
        )
        self._last_actuator_ctrls = np.zeros(len(self._all_actuator_names), dtype=np.float32)

        self.pelvis_link = model.get_link("pelvis")
        self.torso_link = model.get_link("torso_link")
        self.floating_base = model.floating_bases[0] if len(model.floating_bases) else None

    @staticmethod
    def _actuator_name_from_joint(joint_name: str) -> str:
        """Convert a scalar joint name to its matching actuator name."""

        return joint_name.removesuffix("_joint")

    @staticmethod
    def xyzw_to_wxyz(quaternion_xyzw: Sequence[float]) -> np.ndarray:
        """Convert MotrixSim/scipy quaternion order to Unitree/MuJoCo order."""

        q = np.asarray(quaternion_xyzw, dtype=np.float64)
        return q[[3, 0, 1, 2]]

    def link_pose_wxyz(self, link) -> np.ndarray:
        """Return a link pose as ``[x, y, z, qw, qx, qy, qz]``."""

        pose = np.asarray(link.get_pose(self._data), dtype=np.float64)
        return np.concatenate([pose[:3], self.xyzw_to_wxyz(pose[3:7])])

    def bind_data(self, data) -> None:
        """Bind the current SceneData for link helper calls."""

        self._data = data

    def joint_positions(self, data, names: Sequence[str]) -> np.ndarray:
        """Read scalar joint positions in the requested order."""

        dof_positions = np.asarray(self.body.get_joint_dof_pos(data), dtype=np.float64)
        indices = np.asarray([self._joint_name_to_dof_index[name] for name in names], dtype=np.int64)
        return dof_positions[indices]

    def joint_velocities(self, data, names: Sequence[str]) -> np.ndarray:
        """Read scalar joint velocities in the requested order."""

        dof_velocities = np.asarray(self.body.get_joint_dof_vel(data), dtype=np.float64)
        indices = np.asarray([self._joint_name_to_dof_index[name] for name in names], dtype=np.int64)
        return dof_velocities[indices]

    def body_q(self, data) -> np.ndarray:
        """Return 29 body joint positions in MuJoCo simulator order."""

        return np.asarray(self.body.get_joint_dof_pos(data), dtype=np.float64)[self._body_joint_dof_indices]

    def body_dq(self, data) -> np.ndarray:
        """Return 29 body joint velocities in MuJoCo simulator order."""

        return np.asarray(self.body.get_joint_dof_vel(data), dtype=np.float64)[self._body_joint_dof_indices]

    def left_hand_q(self, data) -> np.ndarray:
        """Return left Dex3 joint positions."""

        return np.asarray(self.body.get_joint_dof_pos(data), dtype=np.float64)[
            self._left_hand_joint_dof_indices
        ]

    def left_hand_dq(self, data) -> np.ndarray:
        """Return left Dex3 joint velocities."""

        return np.asarray(self.body.get_joint_dof_vel(data), dtype=np.float64)[
            self._left_hand_joint_dof_indices
        ]

    def right_hand_q(self, data) -> np.ndarray:
        """Return right Dex3 joint positions."""

        return np.asarray(self.body.get_joint_dof_pos(data), dtype=np.float64)[
            self._right_hand_joint_dof_indices
        ]

    def right_hand_dq(self, data) -> np.ndarray:
        """Return right Dex3 joint velocities."""

        return np.asarray(self.body.get_joint_dof_vel(data), dtype=np.float64)[
            self._right_hand_joint_dof_indices
        ]

    def floating_base_pose_wxyz(self, data) -> np.ndarray:
        """Return floating-base pose as ``[x, y, z, qw, qx, qy, qz]``."""

        if self.floating_base is not None:
            translation = np.asarray(self.floating_base.get_translation(data), dtype=np.float64)
            rotation_xyzw = np.asarray(self.floating_base.get_rotation(data), dtype=np.float64)
            return np.concatenate([translation, self.xyzw_to_wxyz(rotation_xyzw)])
        pose = np.asarray(self.pelvis_link.get_pose(data), dtype=np.float64)
        return np.concatenate([pose[:3], self.xyzw_to_wxyz(pose[3:7])])

    def floating_base_velocity(self, data) -> np.ndarray:
        """Return base velocity as ``[global_linear_xyz, local_angular_xyz]``.

        The Unitree low-state publisher uses the linear part for odometry and
        the angular part as an IMU gyroscope reading. MuJoCo's free-joint qvel
        is compatible with this mixed contract in the existing backend, so the
        MotrixSim backend explicitly uses global linear velocity and local
        angular velocity.
        """

        if self.floating_base is not None:
            linear = np.asarray(self.floating_base.get_global_linear_velocity(data), dtype=np.float64)
            angular = np.asarray(self.floating_base.get_local_angular_velocity(data), dtype=np.float64)
        else:
            linear = np.asarray(self.pelvis_link.get_linear_velocity(data), dtype=np.float64)
            angular = self.pelvis_link.get_rotation_mat(data).T @ np.asarray(
                self.pelvis_link.get_angular_velocity(data), dtype=np.float64
            )
        return np.concatenate([linear, angular])

    def torso_pose_wxyz(self, data) -> np.ndarray:
        """Return torso pose as ``[x, y, z, qw, qx, qy, qz]``."""

        pose = np.asarray(self.torso_link.get_pose(data), dtype=np.float64)
        return np.concatenate([pose[:3], self.xyzw_to_wxyz(pose[3:7])])

    def torso_velocity(self, data) -> np.ndarray:
        """Return torso velocity as ``[global_linear_xyz, local_angular_xyz]``."""

        rotation_world_from_torso = self.torso_link.get_rotation_mat(data)
        angular_world = np.asarray(self.torso_link.get_angular_velocity(data), dtype=np.float64)
        return np.concatenate(
            [
                np.asarray(self.torso_link.get_linear_velocity(data), dtype=np.float64),
                rotation_world_from_torso.T @ angular_world,
            ]
        )

    def set_actuator_ctrls(
        self,
        data,
        body_ctrls: np.ndarray,
        left_hand_ctrls: np.ndarray,
        right_hand_ctrls: np.ndarray,
    ) -> None:
        """Set MotrixSim actuator controls in model actuator order."""

        command_by_name = {}
        command_by_name.update(zip(self.body_actuator_names, body_ctrls))
        command_by_name.update(zip(self.left_hand_actuator_names, left_hand_ctrls))
        command_by_name.update(zip(self.right_hand_actuator_names, right_hand_ctrls))
        controls = np.asarray(
            [command_by_name.get(name, 0.0) for name in self._all_actuator_names],
            dtype=np.float32,
        )
        self.body.set_actuator_ctrls(data, controls)
        self._last_actuator_ctrls = controls

    def last_body_torques(self) -> np.ndarray:
        """Return last commanded body torques in body joint order."""

        torque_by_name = dict(zip(self._all_actuator_names, self._last_actuator_ctrls))
        return np.asarray([torque_by_name[name] for name in self.body_actuator_names])

    def last_left_hand_torques(self) -> np.ndarray:
        """Return last commanded left hand torques."""

        torque_by_name = dict(zip(self._all_actuator_names, self._last_actuator_ctrls))
        return np.asarray([torque_by_name[name] for name in self.left_hand_actuator_names])

    def last_right_hand_torques(self) -> np.ndarray:
        """Return last commanded right hand torques."""

        torque_by_name = dict(zip(self._all_actuator_names, self._last_actuator_ctrls))
        return np.asarray([torque_by_name[name] for name in self.right_hand_actuator_names])
