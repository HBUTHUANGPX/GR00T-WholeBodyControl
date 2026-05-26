"""MotrixSim simulation environment and loop for the G1 robot."""

from __future__ import annotations

import pathlib
import pickle
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, Optional

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.motrixsim_sim.human_reference import (
    DeployReferenceFrameSubscriber,
    HumanReferenceConfig,
    HumanReferencePlayer,
)
from gear_sonic.utils.motrixsim_sim.robot import MotrixG1Robot
from gear_sonic.utils.mujoco_sim.robot import Robot
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge


GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class DefaultEnv:
    """MotrixSim environment compatible with the MuJoCo simulator contract.

    Responsibilities:
        Own the MotrixSim model/data, compute PD controls from Unitree SDK
        commands, step physics, publish observations, and sync GUI rendering.
    Preconditions:
        ``motrixsim`` is installed in the active Python environment and the
        configured Gear Sonic G1 scene is loadable by MotrixSim.
    Postconditions:
        ``BaseSimulator`` can drive this environment with the same high-level
        loop used by the MuJoCo backend.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        env_name: str = "default",
        camera_configs: Optional[Dict[str, Any]] = None,
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ) -> None:
        """Create the MotrixSim default environment."""

        self.config = config
        self.env_name = env_name
        self.camera_configs = camera_configs or {}
        self.sim_dt = float(config["SIMULATE_DT"])
        self.viewer_dt = float(config.get("VIEWER_DT", 0.02))
        self.image_dt = float(config.get("IMAGE_DT", 0.033333))
        self.onscreen = bool(onscreen)
        self.offscreen = bool(offscreen)
        self.enable_image_publish = bool(enable_image_publish)
        self.image_publish_process = None
        self.unitree_bridge = None
        self.reward_lock = Lock()
        self.last_reward = 0.0
        self.time = 0.0
        self._warned_image_publish = False
        self._render_context = None
        self.viewer = None
        self.fall = False

        self.legacy_robot_config = Robot(self.config)
        self.torque_limit = np.asarray(self.legacy_robot_config.MOTOR_EFFORT_LIMIT_LIST, dtype=np.float32)

        self._init_scene()
        self._init_human_reference()

    def _init_scene(self) -> None:
        """Load the MotrixSim scene and GUI renderer."""

        import motrixsim as mx
        from motrixsim.render import RenderApp

        xml_path = pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"]
        self.model = mx.load_model(str(xml_path))
        self.model.options.timestep = self.sim_dt
        self.data = mx.SceneData(self.model)
        self.robot = MotrixG1Robot(self.model, self.config)

        if self.onscreen:
            self._render_context = RenderApp("warn")
            self.viewer = self._render_context.__enter__()
            self.viewer.launch(self.model)
            self._configure_camera()
        else:
            self.viewer = None

        if self.offscreen or self.enable_image_publish:
            print(
                "[MotrixSim] Offscreen/image publishing requested. GUI simulation is supported; "
                "framebuffer image publishing is not implemented for MotrixSim yet."
            )

    def _configure_camera(self) -> None:
        """Configure a simple MotrixSim tracking camera when available."""

        try:
            camera_mjcf = """<mujoco model="motrixsim_camera">
  <worldbody>
    <camera name="follower" pos="-1.8 0 1.0" xyaxes="0 -1 0 0 0 1"
      trackposspeed="2" trackrotspeed="2" />
  </worldbody>
</mujoco>"""
            # Runtime camera insertion is not supported after model loading, so
            # this method currently keeps the default MotrixSim camera.
            _ = camera_mjcf
        except Exception:
            pass

    def _init_human_reference(self) -> None:
        """Load optional human skeleton visualization reference."""

        reference_path = self.config.get("MOTRIXSIM_HUMAN_REFERENCE_PATH")
        if reference_path is None:
            default_path = GEAR_SONIC_ROOT / "gear_sonic_deploy" / "reference" / "offline_smpl"
            reference_path = str(default_path) if default_path.exists() else None
        config = HumanReferenceConfig(
            reference_path=Path(reference_path) if reference_path else None,
            enabled=bool(self.config.get("MOTRIXSIM_SHOW_HUMAN_REFERENCE", True)),
            fps=float(self.config.get("MOTRIXSIM_HUMAN_REFERENCE_FPS", 50.0)),
            lateral_offset=float(self.config.get("MOTRIXSIM_HUMAN_REFERENCE_LATERAL_OFFSET", 1.25)),
            forward_offset=float(self.config.get("MOTRIXSIM_HUMAN_REFERENCE_FORWARD_OFFSET", 0.0)),
        )
        self.human_reference = HumanReferencePlayer.from_config(config)
        self.human_reference_frame_subscriber = None
        if self.human_reference is not None and bool(
            self.config.get("MOTRIXSIM_HUMAN_REFERENCE_SYNC_ZMQ", True)
        ):
            self.human_reference_frame_subscriber = DeployReferenceFrameSubscriber(
                host=str(self.config.get("MOTRIXSIM_HUMAN_REFERENCE_SYNC_HOST", "localhost")),
                port=int(self.config.get("MOTRIXSIM_HUMAN_REFERENCE_SYNC_PORT", 5557)),
                topic=str(self.config.get("MOTRIXSIM_HUMAN_REFERENCE_SYNC_TOPIC", "g1_debug")),
            )

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555) -> None:
        """Keep image publishing API-compatible with MuJoCo backend."""

        _ = (start_method, camera_port)
        if not self._warned_image_publish:
            print("[MotrixSim] Image publishing is not implemented; continuing with GUI rendering.")
            self._warned_image_publish = True

    def set_unitree_bridge(self, unitree_bridge) -> None:
        """Attach the Unitree SDK bridge used for command/state transport."""

        self.unitree_bridge = unitree_bridge

    def _low_cmd_available(self) -> bool:
        return self.unitree_bridge is not None and self.unitree_bridge.low_cmd is not None

    def compute_body_torques(self) -> np.ndarray:
        """Compute 29 body joint torques from the latest Unitree LowCmd."""

        torques = np.zeros(self.robot.num_body_joints, dtype=np.float32)
        if not self._low_cmd_available():
            return torques
        q = self.robot.body_q(self.data)
        dq = self.robot.body_dq(self.data)
        with self.unitree_bridge.low_cmd_lock:
            for i in range(self.unitree_bridge.num_body_motor):
                cmd = self.unitree_bridge.low_cmd.motor_cmd[i]
                torques[i] = cmd.tau + cmd.kp * (cmd.q - q[i]) + cmd.kd * (cmd.dq - dq[i])
        return torques

    def compute_hand_torques(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute left/right hand torques from Dex3 commands."""

        left = np.zeros(self.robot.num_hand_joints, dtype=np.float32)
        right = np.zeros(self.robot.num_hand_joints, dtype=np.float32)
        if self.unitree_bridge is None or self.robot.num_hand_joints == 0:
            return left, right
        left_q = self.robot.left_hand_q(self.data)
        left_dq = self.robot.left_hand_dq(self.data)
        right_q = self.robot.right_hand_q(self.data)
        right_dq = self.robot.right_hand_dq(self.data)
        with self.unitree_bridge.left_hand_cmd_lock:
            for i in range(self.unitree_bridge.num_hand_motor):
                cmd = self.unitree_bridge.left_hand_cmd.motor_cmd[i]
                left[i] = cmd.tau + cmd.kp * (cmd.q - left_q[i]) + cmd.kd * (cmd.dq - left_dq[i])
        with self.unitree_bridge.right_hand_cmd_lock:
            for i in range(self.unitree_bridge.num_hand_motor):
                cmd = self.unitree_bridge.right_hand_cmd.motor_cmd[i]
                right[i] = cmd.tau + cmd.kp * (cmd.q - right_q[i]) + cmd.kd * (cmd.dq - right_dq[i])
        return left, right

    def _clip_actuator_torques(
        self,
        body_torques: np.ndarray,
        left_hand_torques: np.ndarray,
        right_hand_torques: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Clip controls with the configured actuator effort limits."""

        command_by_name = {}
        command_by_name.update(zip(self.robot.body_actuator_names, body_torques))
        command_by_name.update(zip(self.robot.left_hand_actuator_names, left_hand_torques))
        command_by_name.update(zip(self.robot.right_hand_actuator_names, right_hand_torques))
        all_controls = np.asarray(
            [command_by_name.get(name, 0.0) for name in self.model.actuator_names],
            dtype=np.float32,
        )
        if len(self.torque_limit) == len(all_controls):
            all_controls = np.clip(all_controls, -self.torque_limit, self.torque_limit)
        clipped = dict(zip(self.model.actuator_names, all_controls))
        return (
            np.asarray([clipped[name] for name in self.robot.body_actuator_names], dtype=np.float32),
            np.asarray([clipped[name] for name in self.robot.left_hand_actuator_names], dtype=np.float32),
            np.asarray([clipped[name] for name in self.robot.right_hand_actuator_names], dtype=np.float32),
        )

    def prepare_obs(self) -> Dict[str, Any]:
        """Create the observation dictionary consumed by UnitreeSdk2Bridge."""

        base_pose = self.robot.floating_base_pose_wxyz(self.data)
        base_vel = self.robot.floating_base_velocity(self.data)
        torso_pose = self.robot.torso_pose_wxyz(self.data)
        torso_vel = self.robot.torso_velocity(self.data)
        body_q = self.robot.body_q(self.data)
        body_dq = self.robot.body_dq(self.data)
        left_hand_q = self.robot.left_hand_q(self.data)
        left_hand_dq = self.robot.left_hand_dq(self.data)
        right_hand_q = self.robot.right_hand_q(self.data)
        right_hand_dq = self.robot.right_hand_dq(self.data)

        return {
            "floating_base_pose": base_pose,
            "floating_base_vel": base_vel,
            "floating_base_acc": np.zeros(6),
            "secondary_imu_quat": torso_pose[3:7],
            "secondary_imu_vel": torso_vel,
            "body_q": body_q,
            "body_dq": body_dq,
            "body_ddq": np.zeros_like(body_q),
            "body_tau_est": self.robot.last_body_torques(),
            "left_hand_q": left_hand_q,
            "left_hand_dq": left_hand_dq,
            "left_hand_ddq": np.zeros_like(left_hand_q),
            "left_hand_tau_est": self.robot.last_left_hand_torques(),
            "right_hand_q": right_hand_q,
            "right_hand_dq": right_hand_dq,
            "right_hand_ddq": np.zeros_like(right_hand_q),
            "right_hand_tau_est": self.robot.last_right_hand_torques(),
            "time": self.time,
        }

    def sim_step(self) -> None:
        """Publish state, apply controls, and step MotrixSim physics once."""

        obs = self.prepare_obs()
        if self.unitree_bridge is not None:
            self.unitree_bridge.PublishLowState(obs)
            if self.unitree_bridge.joystick:
                self.unitree_bridge.PublishWirelessController()

        body_torques = self.compute_body_torques()
        left_hand_torques, right_hand_torques = self.compute_hand_torques()
        body_torques, left_hand_torques, right_hand_torques = self._clip_actuator_torques(
            body_torques, left_hand_torques, right_hand_torques
        )
        self.robot.set_actuator_ctrls(self.data, body_torques, left_hand_torques, right_hand_torques)
        self.model.step(self.data)
        self.time += self.sim_dt
        self.check_fall()

    def get_head_pose(self) -> np.ndarray:
        """Return a head/torso proxy pose used by optional redis publishing."""

        torso_pose = self.robot.torso_pose_wxyz(self.data)
        # Redis users historically expect xyzw after position.
        return np.concatenate([torso_pose[:3], torso_pose[[4, 5, 6, 3]]])

    def update_viewer(self) -> None:
        """Sync GUI rendering and draw optional human reference skeleton."""

        if self.viewer is None:
            return
        if self.human_reference is not None:
            frame_index = None
            if self.human_reference_frame_subscriber is not None:
                frame_index = self.human_reference_frame_subscriber.poll_latest_frame()
                if frame_index is None:
                    frame_index = 0
            self.human_reference.draw(self.viewer.gizmos, self.time, frame_index=frame_index)
        self.viewer.sync(self.data)

    def update_reward(self) -> None:
        """Update reward placeholder."""

        with self.reward_lock:
            self.last_reward = 0.0

    def get_reward(self) -> float:
        """Return the latest reward placeholder."""

        with self.reward_lock:
            return self.last_reward

    def update_render_caches(self) -> Dict[str, Any]:
        """Return render caches for image publishing.

        MotrixSim GUI rendering is supported. Offscreen framebuffer extraction is
        intentionally left as a future extension because the current public API
        does not expose a stable image readback path in the examples.
        """

        if (self.offscreen or self.enable_image_publish) and not self._warned_image_publish:
            print("[MotrixSim] Offscreen image readback is not implemented.")
            self._warned_image_publish = True
        return {}

    def handle_keyboard_button(self, key: str) -> None:
        """Handle simulator-level keyboard commands."""

        if key == "backspace":
            self.reset()

    def check_fall(self) -> None:
        """Reset if the pelvis falls below a conservative threshold."""

        base_pose = self.robot.floating_base_pose_wxyz(self.data)
        self.fall = bool(base_pose[2] < 0.2)
        if self.fall:
            print(f"Warning: Robot has fallen in MotrixSim, height: {base_pose[2]:.3f} m")
            self.reset()

    def reset(self) -> None:
        """Reset MotrixSim data to the model default state."""

        import motrixsim as mx

        self.data = mx.SceneData(self.model)
        self.robot.bind_data(self.data)
        self.time = 0.0

    def close(self) -> None:
        """Close the MotrixSim render context if it was opened."""

        if self._render_context is not None:
            try:
                self._render_context.__exit__(None, None, None)
            finally:
                self._render_context = None
                self.viewer = None
        if self.human_reference_frame_subscriber is not None:
            self.human_reference_frame_subscriber.close()
            self.human_reference_frame_subscriber = None

    def get_privileged_obs(self) -> Dict[str, Any]:
        """Return privileged observations placeholder."""

        return {}


class BaseSimulator:
    """MotrixSim simulator wrapper matching the MuJoCo BaseSimulator API."""

    def __init__(
        self,
        config: Dict[str, Any],
        env_name: str = "default",
        redis_client=None,
        **kwargs,
    ) -> None:
        """Create the MotrixSim simulator wrapper."""

        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        self.sim_dt = float(self.config["SIMULATE_DT"])
        self.reward_dt = float(self.config.get("REWARD_DT", 0.02))
        self.image_dt = float(self.config.get("IMAGE_DT", 0.033333))
        self.viewer_dt = float(self.config.get("VIEWER_DT", 0.02))
        self._running = True
        self.sim_thread = None

        if env_name != "default":
            raise ValueError(
                f"Invalid environment name: {env_name}. Only 'default' is supported for MotrixSim."
            )
        self.sim_env = DefaultEnv(config, env_name, **kwargs)

        try:
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as exc:
            print(f"Note: Channel factory initialization attempt: {exc}")

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)
        self.init_subscriber()
        self.init_publisher()

    def start_as_thread(self) -> None:
        """Start simulation on a background thread."""

        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555) -> None:
        """Forward image publish startup to the environment."""

        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self) -> None:
        """Compatibility extension point."""

    def init_publisher(self) -> None:
        """Compatibility extension point."""

    def init_unitree_bridge(self) -> None:
        """Create the Unitree SDK bridge."""

        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self) -> None:
        """Main MotrixSim loop with physics, GUI, reward, and image rates."""

        sim_cnt = 0
        ts = time.time()
        try:
            while self._running and (
                (self.sim_env.viewer is not None and not self.sim_env.viewer.is_closed)
                or self.sim_env.viewer is None
            ):
                step_start = time.monotonic()
                self.sim_env.sim_step()

                now = time.time()
                if now - ts > 0.1 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                if sim_cnt % max(1, int(round(self.viewer_dt / self.sim_dt))) == 0:
                    self.sim_env.update_viewer()
                if sim_cnt % max(1, int(round(self.reward_dt / self.sim_dt))) == 0:
                    self.sim_env.update_reward()
                if sim_cnt % max(1, int(round(self.image_dt / self.sim_dt))) == 0:
                    self.sim_env.update_render_caches()

                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                sim_cnt += 1
        except KeyboardInterrupt:
            print("MotrixSim simulator interrupted by user.")
        finally:
            self.close()

    def __del__(self) -> None:
        self.close()

    def reset(self) -> None:
        """Reset the simulation."""

        self.sim_env.reset()

    def close(self) -> None:
        """Stop the simulation and close resources."""

        self._running = False
        try:
            self.sim_env.close()
        except Exception as exc:
            print(f"Warning during MotrixSim close: {exc}")

    def get_privileged_obs(self) -> Dict[str, Any]:
        """Return privileged observations."""

        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key: str) -> None:
        """Forward simulator keyboard events."""

        self.sim_env.handle_keyboard_button(key)
