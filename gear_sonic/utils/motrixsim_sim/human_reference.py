"""Offline SMPL reference visualization helpers for the MotrixSim backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import msgpack
import numpy as np
import zmq


SMPL_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (21, 23),
)


@dataclass(frozen=True)
class HumanReferenceConfig:
    """Configuration for MotrixSim human skeleton visualization.

    Preconditions:
        ``reference_path`` is ``None``, a motion directory containing
        ``smpl_joint.csv``, or a root directory containing one or more motion
        directories.
    Postconditions:
        ``HumanReferencePlayer.from_config`` can create a player or return
        ``None`` when visualization is disabled or unavailable.
    """

    reference_path: Optional[Path]
    enabled: bool = True
    fps: float = 50.0
    lateral_offset: float = 1.25
    forward_offset: float = 0.0
    ground_clearance: float = 0.02


class HumanReferencePlayer:
    """Read and draw a deploy-format SMPL reference motion.

    Responsibilities:
        Load ``smpl_joint.csv`` from a deploy reference folder and render the
        current skeleton frame with MotrixSim gizmos.
    Preconditions:
        The reference CSV stores 24 or more SMPL joints as ``J * 3`` columns.
    Postconditions:
        ``draw`` emits transient spheres and lines for the current frame.
    """

    def __init__(self, joint_frames: np.ndarray, config: HumanReferenceConfig) -> None:
        """Create a human reference player.

        Preconditions:
            ``joint_frames`` has shape ``(T, J, 3)`` with finite values.
        Postconditions:
            The player can sample frames by simulation time.
        """

        if joint_frames.ndim != 3 or joint_frames.shape[1] < 24 or joint_frames.shape[2] != 3:
            raise ValueError("SMPL joint frames must have shape (T, at least 24, 3)")
        if len(joint_frames) == 0:
            raise ValueError("SMPL joint reference is empty")
        if not np.isfinite(joint_frames).all():
            raise ValueError("SMPL joint reference contains non-finite values")
        if config.fps <= 0.0:
            raise ValueError("Human reference fps must be positive")
        self._frames = joint_frames.astype(np.float32, copy=False)
        self._config = config
        self._warned_draw_failure = False

    @classmethod
    def from_config(cls, config: HumanReferenceConfig) -> Optional["HumanReferencePlayer"]:
        """Build a player from configuration.

        Preconditions:
            ``config`` contains a candidate reference path.
        Postconditions:
            Returns ``None`` when disabled or no reference can be found.
        """

        if not config.enabled or config.reference_path is None:
            return None
        motion_dir = cls._resolve_motion_dir(config.reference_path)
        if motion_dir is None:
            print(f"[MotrixSim] Human reference not found: {config.reference_path}")
            return None
        csv_path = motion_dir / "smpl_joint.csv"
        joint_rows = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if joint_rows.ndim == 1:
            joint_rows = joint_rows[None, :]
        if joint_rows.shape[1] % 3 != 0:
            raise ValueError(f"{csv_path} column count must be divisible by 3")
        joint_frames = joint_rows.reshape(joint_rows.shape[0], joint_rows.shape[1] // 3, 3)
        print(f"[MotrixSim] Loaded human reference skeleton: {motion_dir} ({len(joint_frames)} frames)")
        return cls(joint_frames, config)

    @staticmethod
    def _resolve_motion_dir(path: Path) -> Optional[Path]:
        """Resolve a reference root or motion folder to a motion folder.

        Preconditions:
            ``path`` may or may not exist.
        Postconditions:
            Returns a directory containing ``smpl_joint.csv`` or ``None``.
        """

        path = Path(path)
        if (path / "smpl_joint.csv").exists():
            return path
        if not path.exists() or not path.is_dir():
            return None
        candidates = sorted(child for child in path.iterdir() if (child / "smpl_joint.csv").exists())
        return candidates[0] if candidates else None

    def frame_at_time(self, sim_time: float) -> np.ndarray:
        """Return the skeleton frame for a simulation timestamp.

        Preconditions:
            ``sim_time`` is measured in seconds.
        Postconditions:
            Returns one ``(J, 3)`` array, looping over the reference.
        """

        frame_index = int(max(0.0, sim_time) * self._config.fps) % len(self._frames)
        points = self._frames[frame_index].copy()
        points[:, 0] += self._config.forward_offset
        points[:, 1] += self._config.lateral_offset
        points[:, 2] += self._config.ground_clearance - float(points[:, 2].min())
        return points

    def frame_at_index(self, frame_index: int) -> np.ndarray:
        """Return the skeleton frame for an explicit deploy reference frame.

        Preconditions:
            ``frame_index`` is produced by the deploy current-motion cursor.
        Postconditions:
            Returns one ``(J, 3)`` array with visualization offsets applied.
        """

        safe_index = max(0, int(frame_index)) % len(self._frames)
        points = self._frames[safe_index].copy()
        points[:, 0] += self._config.forward_offset
        points[:, 1] += self._config.lateral_offset
        points[:, 2] += self._config.ground_clearance - float(points[:, 2].min())
        return points

    def draw(self, gizmos, sim_time: float, frame_index: Optional[int] = None) -> None:
        """Draw the current skeleton frame with MotrixSim gizmos.

        Preconditions:
            ``gizmos`` is a ``motrixsim.render.RenderGizmos`` instance.
        Postconditions:
            The skeleton is queued for the current render frame. Rendering
            failures are reported once and then ignored.
        """

        try:
            from motrixsim.render import Color

            points = self.frame_at_index(frame_index) if frame_index is not None else self.frame_at_time(sim_time)
            joint_color = Color.rgb(0.1, 0.8, 1.0)
            bone_color = Color.rgb(1.0, 0.75, 0.15)
            for start, end in SMPL_BONE_EDGES:
                gizmos.draw_line(points[start], points[end], color=bone_color)
            for point in points[:24]:
                gizmos.draw_sphere(0.025, point, color=joint_color)
        except Exception as exc:
            if not self._warned_draw_failure:
                print(f"[MotrixSim] Human skeleton draw disabled after error: {exc}")
                self._warned_draw_failure = True


class DeployReferenceFrameSubscriber:
    """Subscribe to deploy's ZMQ debug stream and cache the latest motion frame.

    Responsibilities:
        Keep MotrixSim-only visualization synchronized to the frame cursor used
        by ``g1_deploy_onnx_ref``.
    Preconditions:
        Deploy is started with ``--output-type zmq`` and publishes the
        ``current_motion_frame`` field on the configured topic.
    Postconditions:
        ``poll_latest_frame`` returns the newest non-negative deploy frame, or
        ``None`` until one is received.
    """

    def __init__(self, host: str = "localhost", port: int = 5557, topic: str = "g1_debug") -> None:
        """Create a non-blocking ZMQ subscriber for deploy visualization state."""

        self._topic = topic
        self._topic_bytes = topic.encode("utf-8")
        self._latest_frame: Optional[int] = None
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.connect(f"tcp://{host}:{port}")
        print(f"[MotrixSim] Human reference sync connected to tcp://{host}:{port} ({topic})")

    def poll_latest_frame(self) -> Optional[int]:
        """Poll deploy ZMQ once and return the latest cached reference frame."""

        if self._socket is None:
            return self._latest_frame
        while True:
            try:
                raw = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                return self._latest_frame
            except Exception as exc:
                print(f"[MotrixSim] Human reference sync disabled after ZMQ error: {exc}")
                self.close()
                return self._latest_frame

            if not raw.startswith(self._topic_bytes):
                continue
            payload = raw[len(self._topic_bytes) :]
            try:
                message = msgpack.unpackb(payload, raw=False)
            except Exception:
                continue
            frame_value = message.get("current_motion_frame")
            if isinstance(frame_value, list) and frame_value:
                frame_value = frame_value[0]
            if frame_value is None:
                continue
            frame = int(frame_value)
            if frame >= 0:
                self._latest_frame = frame

    def close(self) -> None:
        """Close the ZMQ subscriber resources."""

        socket = getattr(self, "_socket", None)
        if socket is not None:
            socket.close(linger=0)
            self._socket = None
        ctx = getattr(self, "_ctx", None)
        if ctx is not None:
            ctx.term()
            self._ctx = None
