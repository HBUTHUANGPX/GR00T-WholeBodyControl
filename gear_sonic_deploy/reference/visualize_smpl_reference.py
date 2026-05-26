#!/usr/bin/env python3
"""Visualize SMPL skeleton data from an offline npz or deploy reference folder.

This module intentionally keeps data loading, validation, SMPL inference, and
rendering separate.  The split makes it easier to inspect coordinate-frame bugs
without tying the tool to MuJoCo, ZMQ, or the deploy controller runtime.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
from xml.sax.saxutils import escape

import numpy as np


SMPL_JOINT_EDGES: tuple[tuple[int, int], ...] = (
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
    (12, 13),
    (12, 14),
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
class SkeletonFrame:
    """One visualizable SMPL skeleton frame.

    Responsibilities:
        Store joint positions and optional root orientation for a single frame.
    Preconditions:
        ``joint_positions`` has shape ``(J, 3)`` and finite numeric values.
        ``root_quaternion_wxyz`` is ``None`` or a finite ``(4,)`` quaternion in
        scalar-first order.
    Postconditions:
        Instances are immutable containers for downstream renderers.
    """

    joint_positions: np.ndarray
    root_quaternion_wxyz: Optional[np.ndarray]
    frame_index: int
    label: str


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for skeleton loading and display.

    Responsibilities:
        Collect CLI tunables in one immutable object.
    Preconditions:
        ``start_frame >= 0``, ``stride > 0``, and ``max_frames`` is ``None`` or
        positive.
    Postconditions:
        Loaders can use the same frame slicing policy consistently.
    """

    start_frame: int = 0
    max_frames: Optional[int] = None
    stride: int = 1
    show_root_axes: bool = True
    axis_length: float = 0.25
    output_image: Optional[Path] = None


@dataclass(frozen=True)
class MujocoSceneConfig:
    """Configuration for static MuJoCo skeleton scenes.

    Responsibilities:
        Store visual sizes and world-axis conventions for MJCF export.
    Preconditions:
        Radii and axis length are positive.
    Postconditions:
        ``MujocoSkeletonSceneBuilder`` can render skeletons consistently.
    """

    joint_radius: float = 0.025
    bone_radius: float = 0.01
    axis_radius: float = 0.015
    axis_length: float = 0.75
    add_floor: bool = True


class SkeletonSource(ABC):
    """Abstract interface for any source that can provide SMPL skeleton frames.

    Responsibilities:
        Decouple visualization from storage format.
    Preconditions:
        Implementations validate their own file paths and dependencies.
    Postconditions:
        ``load`` returns frames ordered by playback time.
    """

    @abstractmethod
    def load(self) -> list[SkeletonFrame]:
        """Load skeleton frames.

        Preconditions:
            Source-specific paths and optional models are available.
        Postconditions:
            Returns a non-empty list of finite ``SkeletonFrame`` objects.
        Raises:
            FileNotFoundError: Required source files are missing.
            ValueError: Source data has invalid shape or contains no frames.
        """


class CsvArrayReader:
    """Read numeric deploy CSV files with one header row.

    Responsibilities:
        Provide a tiny, testable CSV adapter for reference folders.
    Preconditions:
        Files are comma-separated and include exactly one header row.
    Postconditions:
        Returned arrays are at least two-dimensional and dtype ``float64``.
    """

    def read(self, path: Path) -> np.ndarray:
        """Read a CSV file into an array.

        Preconditions:
            ``path`` exists and contains numeric rows after the header.
        Postconditions:
            Returns an array with shape ``(frames, columns)``.
        """

        if not path.exists():
            raise FileNotFoundError(f"Required CSV file not found: {path}")
        array = np.loadtxt(path, delimiter=",", skiprows=1)
        if array.ndim == 1:
            array = array[None, :]
        if array.size == 0:
            raise ValueError(f"CSV file has no data rows: {path}")
        if not np.isfinite(array).all():
            raise ValueError(f"CSV file contains non-finite values: {path}")
        return array


class FrameSlicer:
    """Apply a shared start/max/stride policy to frame-indexed arrays.

    Responsibilities:
        Keep slicing behavior identical across npz and reference sources.
    Preconditions:
        Arrays use frame as the first dimension.
    Postconditions:
        Returned arrays retain all non-frame dimensions.
    """

    def __init__(self, config: VisualizationConfig) -> None:
        """Create a slicer.

        Preconditions:
            ``config`` has valid frame slicing fields.
        Postconditions:
            The slicer can be reused for multiple arrays with equal frame count.
        """

        if config.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if config.stride <= 0:
            raise ValueError("stride must be positive")
        if config.max_frames is not None and config.max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        self._config = config

    def slice(self, array: np.ndarray) -> np.ndarray:
        """Slice frames from an array.

        Preconditions:
            ``array`` has at least one dimension.
        Postconditions:
            Returns frames according to ``VisualizationConfig``.
        """

        end = None
        if self._config.max_frames is not None:
            end = self._config.start_frame + self._config.max_frames * self._config.stride
        return np.asarray(array)[self._config.start_frame : end : self._config.stride]


class ReferenceSkeletonSource(SkeletonSource):
    """Load SMPL skeleton frames from a deploy reference motion directory.

    Responsibilities:
        Read ``smpl_joint.csv`` and optional ``body_quat.csv`` from one motion
        folder and expose them as visualizable frames.
    Preconditions:
        The directory contains ``smpl_joint.csv`` with 24*3 columns.
    Postconditions:
        Each frame contains 24 SMPL joint positions and optional root quaternion.
    """

    def __init__(
        self,
        motion_directory: Path,
        config: Optional[VisualizationConfig] = None,
        csv_reader: Optional[CsvArrayReader] = None,
    ) -> None:
        """Create a reference-folder source.

        Preconditions:
            ``motion_directory`` points to a single motion folder, not a parent
            directory containing multiple motions.
        Postconditions:
            ``load`` can read SMPL joints from the directory.
        """

        self._motion_directory = Path(motion_directory)
        self._config = config or VisualizationConfig()
        self._csv_reader = csv_reader or CsvArrayReader()
        self._slicer = FrameSlicer(self._config)

    def load(self) -> list[SkeletonFrame]:
        """Load frames from ``smpl_joint.csv`` and ``body_quat.csv``.

        Preconditions:
            ``smpl_joint.csv`` exists and has a column count divisible by 3.
        Postconditions:
            Returns one frame per selected CSV row.
        """

        joint_rows = self._slicer.slice(self._csv_reader.read(self._motion_directory / "smpl_joint.csv"))
        if joint_rows.shape[1] % 3 != 0:
            raise ValueError("smpl_joint.csv must contain J*3 columns")
        joint_positions = joint_rows.reshape(joint_rows.shape[0], joint_rows.shape[1] // 3, 3)
        if joint_positions.shape[1] < 24:
            raise ValueError("smpl_joint.csv must contain at least 24 joints")
        joint_positions = joint_positions[:, :24, :]

        root_quaternions = None
        body_quat_path = self._motion_directory / "body_quat.csv"
        if body_quat_path.exists():
            quat_rows = self._slicer.slice(self._csv_reader.read(body_quat_path))
            if quat_rows.shape[1] < 4:
                raise ValueError("body_quat.csv must contain at least one wxyz quaternion")
            root_quaternions = quat_rows[:, :4]
            if len(root_quaternions) != len(joint_positions):
                raise ValueError("body_quat.csv and smpl_joint.csv frame counts differ after slicing")

        frames = []
        for frame_offset, joints in enumerate(joint_positions):
            quaternion = None if root_quaternions is None else root_quaternions[frame_offset]
            frames.append(
                SkeletonFrame(
                    joint_positions=joints.astype(np.float32),
                    root_quaternion_wxyz=None if quaternion is None else quaternion.astype(np.float32),
                    frame_index=self._config.start_frame + frame_offset * self._config.stride,
                    label=self._motion_directory.name,
                )
            )
        return frames


class SmplModelRunner:
    """Run SMPL forward kinematics through the ``smplx`` package.

    Responsibilities:
        Isolate the optional torch/smplx dependency from the rest of the tool.
    Preconditions:
        ``smplx`` and ``torch`` are importable and a SMPL model path is provided.
    Postconditions:
        Returns 24 world-space SMPL joint positions for each frame.
    """

    def __init__(self, model_path: Path, batch_size: int = 512) -> None:
        """Create a SMPL model runner.

        Preconditions:
            ``model_path`` points either to a SMPL model directory or to a model
            file inside a ``smpl`` directory.
        Postconditions:
            ``compute_joints`` can run batched SMPL inference.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._model_path = self._normalize_model_path(Path(model_path))
        self._batch_size = batch_size
        self._model = None

    @staticmethod
    def _normalize_model_path(model_path: Path) -> Path:
        """Normalize a SMPL model file or directory path for ``smplx.create``.

        Preconditions:
            ``model_path`` is user supplied and may be relative.
        Postconditions:
            Returns the parent body-model directory expected by ``smplx``.
        """

        resolved = model_path.expanduser().resolve()
        if resolved.is_file() and resolved.parent.name.lower() == "smpl":
            return resolved.parent.parent
        if resolved.name.lower() == "smpl":
            return resolved.parent
        return resolved.parent if resolved.is_file() else resolved

    def compute_joints(
        self,
        global_orient: np.ndarray,
        body_pose: np.ndarray,
        transl: np.ndarray,
        betas: np.ndarray,
    ) -> np.ndarray:
        """Compute SMPL joints.

        Preconditions:
            Arrays have matching frame count; ``body_pose`` has at least 69
            columns; ``global_orient`` and ``transl`` have shape ``(T, 3)``.
        Postconditions:
            Returns finite ``(T, 24, 3)`` joint positions in SMPL model space.
        """

        import torch
        import smplx

        if self._model is None:
            self._model = smplx.create(
                str(self._model_path),
                model_type="smpl",
                gender="neutral",
                batch_size=min(self._batch_size, len(global_orient)),
                num_betas=betas.shape[1],
            )
            self._model.eval()

        joints = []
        with torch.no_grad():
            for start in range(0, len(global_orient), self._batch_size):
                end = min(start + self._batch_size, len(global_orient))
                output = self._model(
                    global_orient=torch.as_tensor(global_orient[start:end], dtype=torch.float32),
                    body_pose=torch.as_tensor(body_pose[start:end, :69], dtype=torch.float32),
                    betas=torch.as_tensor(betas[start:end], dtype=torch.float32),
                    transl=torch.as_tensor(transl[start:end], dtype=torch.float32),
                )
                joints.append(output.joints[:, :24, :].detach().cpu().numpy())
        result = np.concatenate(joints, axis=0).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("SMPL model produced non-finite joints")
        return result


class NpzSmplSkeletonSource(SkeletonSource):
    """Load raw SMPL skeleton frames from an npz file.

    Responsibilities:
        Read ``global_orient``, ``body_pose``, ``transl``, and ``betas`` and
        compute raw SMPL joint positions with a provided model runner.
    Preconditions:
        The npz contains the required SMPL parameter arrays.
    Postconditions:
        Returns visualizable world/model-space skeleton frames.
    """

    def __init__(
        self,
        npz_path: Path,
        model_runner: SmplModelRunner,
        config: Optional[VisualizationConfig] = None,
    ) -> None:
        """Create an npz SMPL source.

        Preconditions:
            ``npz_path`` exists and ``model_runner`` can compute SMPL joints.
        Postconditions:
            ``load`` exposes raw SMPL skeleton frames.
        """

        self._npz_path = Path(npz_path)
        self._model_runner = model_runner
        self._config = config or VisualizationConfig()
        self._slicer = FrameSlicer(self._config)

    def load(self) -> list[SkeletonFrame]:
        """Load and forward SMPL parameters from the npz file.

        Preconditions:
            The npz contains ``global_orient``, ``body_pose``, and ``transl``.
        Postconditions:
            Returns frames with SMPL model-computed joint positions.
        """

        if not self._npz_path.exists():
            raise FileNotFoundError(f"SMPL npz not found: {self._npz_path}")
        with np.load(self._npz_path) as data:
            global_orient = self._slicer.slice(self._require(data, "global_orient")).astype(np.float32)
            body_pose = self._slicer.slice(self._require(data, "body_pose")).astype(np.float32)
            transl = self._slicer.slice(self._require(data, "transl")).astype(np.float32)
            if "betas" in data:
                betas = self._slicer.slice(data["betas"]).astype(np.float32)
            else:
                betas = np.zeros((len(global_orient), 10), dtype=np.float32)

        if betas.ndim == 1:
            betas = np.tile(betas[None, :], (len(global_orient), 1))
        if len(betas) == 1 and len(global_orient) > 1:
            betas = np.tile(betas, (len(global_orient), 1))
        joints = self._model_runner.compute_joints(global_orient, body_pose, transl, betas)
        root_quats = RotationAdapter.axis_angle_to_quat_wxyz(global_orient)
        return [
            SkeletonFrame(joints[idx], root_quats[idx], self._config.start_frame + idx * self._config.stride, self._npz_path.stem)
            for idx in range(len(joints))
        ]

    @staticmethod
    def _require(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
        """Read a required npz array.

        Preconditions:
            ``data`` is an open npz file.
        Postconditions:
            Returns the requested array or raises a clear exception.
        """

        if key not in data:
            raise ValueError(f"Input npz is missing required key '{key}'")
        return np.asarray(data[key])


class RotationAdapter:
    """Small NumPy/SciPy rotation helper.

    Responsibilities:
        Provide scalar-first quaternion utilities without leaking SciPy calls
        through the visualization classes.
    Preconditions:
        Inputs are finite numeric arrays with trailing dimension 3 or 4.
    Postconditions:
        Outputs use scalar-first ``wxyz`` quaternion order.
    """

    @staticmethod
    def axis_angle_to_quat_wxyz(axis_angle: np.ndarray) -> np.ndarray:
        """Convert axis-angle vectors to scalar-first quaternions.

        Preconditions:
            ``axis_angle`` has trailing dimension 3.
        Postconditions:
            Returns an array with trailing dimension 4 in ``wxyz`` order.
        """

        from scipy.spatial.transform import Rotation

        quat_xyzw = Rotation.from_rotvec(np.asarray(axis_angle).reshape(-1, 3)).as_quat()
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        return quat_wxyz.reshape(np.asarray(axis_angle).shape[:-1] + (4,)).astype(np.float32)

    @staticmethod
    def quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
        """Convert a scalar-first quaternion to a rotation matrix.

        Preconditions:
            ``quaternion`` has shape ``(4,)`` and finite values.
        Postconditions:
            Returns a ``(3, 3)`` rotation matrix.
        """

        from scipy.spatial.transform import Rotation

        q = np.asarray(quaternion, dtype=np.float64)
        return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


class SkeletonFrameValidator:
    """Compute sanity-check summaries for skeleton frames.

    Responsibilities:
        Provide numeric checks that help distinguish world-space and root-local
        skeleton data before opening a GUI.
    Preconditions:
        Frames are non-empty and contain finite joint positions.
    Postconditions:
        Returns plain Python values suitable for logging or assertions.
    """

    def summarize(self, frames: Sequence[SkeletonFrame]) -> dict[str, object]:
        """Summarize a sequence of skeleton frames.

        Preconditions:
            ``frames`` is non-empty.
        Postconditions:
            Returns root magnitude, coordinate bounds, and a root-local flag.
        """

        if not frames:
            raise ValueError("Cannot summarize an empty frame list")
        joints = np.stack([frame.joint_positions for frame in frames], axis=0)
        if not np.isfinite(joints).all():
            raise ValueError("Skeleton frames contain non-finite joint positions")
        root_max_abs = float(np.abs(joints[:, 0, :]).max())
        return {
            "num_frames": len(frames),
            "num_joints": int(joints.shape[1]),
            "root_max_abs": root_max_abs,
            "is_root_local": bool(root_max_abs < 1e-5),
            "min_xyz": joints.reshape(-1, 3).min(axis=0).tolist(),
            "max_xyz": joints.reshape(-1, 3).max(axis=0).tolist(),
        }


class CoordinateTransform(ABC):
    """Abstract interface for coordinate-frame transforms.

    Responsibilities:
        Convert skeleton points between source and target world conventions.
    Preconditions:
        Implementations receive finite arrays with trailing dimension 3.
    Postconditions:
        Transformed frames preserve time index and source identity metadata.
    """

    @abstractmethod
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform 3D points.

        Preconditions:
            ``points`` has trailing dimension 3.
        Postconditions:
            Returns transformed points with the same shape.
        """

    def transform_frame(self, frame: SkeletonFrame) -> SkeletonFrame:
        """Transform all joints in a skeleton frame.

        Preconditions:
            ``frame`` contains finite joint positions.
        Postconditions:
            Returns a new frame in the target coordinate convention.
        """

        return SkeletonFrame(
            joint_positions=self.transform_points(frame.joint_positions).astype(np.float32),
            root_quaternion_wxyz=frame.root_quaternion_wxyz,
            frame_index=frame.frame_index,
            label=f"{frame.label}_mujoco",
        )


class IdentityCoordinateTransform(CoordinateTransform):
    """Leave skeleton coordinates unchanged.

    Responsibilities:
        Make raw-coordinate inspection explicit in the CLI.
    Preconditions:
        Input points have trailing dimension 3.
    Postconditions:
        Output points equal input points.
    """

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Return a copy of the input points.

        Preconditions:
            ``points`` has trailing dimension 3.
        Postconditions:
            Returns a float32 copy with unchanged coordinates.
        """

        array = np.asarray(points, dtype=np.float32)
        if array.shape[-1] != 3:
            raise ValueError("points must have trailing dimension 3")
        return array.copy()


class SmplToMujocoCoordinateTransform(CoordinateTransform):
    """Convert common SMPL world axes to robot MuJoCo axes.

    Responsibilities:
        Map SMPL-like ``x-right, y-up, -z-front`` points into the robot
        convention ``x-front, y-left, z-up``.
    Preconditions:
        Input points are expressed in the raw SMPL model/world convention.
    Postconditions:
        Output points are suitable for direct inspection in MuJoCo z-up world.
    """

    _MATRIX = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply ``mujoco = [-smpl_z, -smpl_x, smpl_y]``.

        Preconditions:
            ``points`` has trailing dimension 3.
        Postconditions:
            Returns transformed points with the same shape.
        """

        array = np.asarray(points, dtype=np.float32)
        if array.shape[-1] != 3:
            raise ValueError("points must have trailing dimension 3")
        flat = array.reshape(-1, 3)
        transformed = flat @ self._MATRIX.T
        return transformed.reshape(array.shape)


class GroundAlignmentTransform(CoordinateTransform):
    """Lift a transformed skeleton so its lowest joint is above the floor.

    Responsibilities:
        Decorate another coordinate transform with MuJoCo floor alignment for
        visual inspection.
    Preconditions:
        ``base_transform`` returns finite points and ``clearance`` is
        non-negative.
    Postconditions:
        The minimum output z coordinate equals ``clearance``.
    """

    def __init__(self, base_transform: CoordinateTransform, clearance: float = 0.0) -> None:
        """Create a ground-alignment transform.

        Preconditions:
            ``base_transform`` is a valid coordinate transform and
            ``clearance >= 0``.
        Postconditions:
            ``transform_points`` applies base transform then z translation.
        """

        if clearance < 0:
            raise ValueError("clearance must be non-negative")
        self._base_transform = base_transform
        self._clearance = float(clearance)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform points and lift them to the MuJoCo floor.

        Preconditions:
            ``points`` has trailing dimension 3.
        Postconditions:
            Returns points whose minimum z coordinate is ``clearance``.
        """

        transformed = self._base_transform.transform_points(points).astype(np.float32)
        min_z = float(np.min(transformed[..., 2]))
        aligned = transformed.copy()
        aligned[..., 2] += self._clearance - min_z
        return aligned

    def transform_frame(self, frame: SkeletonFrame) -> SkeletonFrame:
        """Transform a frame and mark it as ground-aligned.

        Preconditions:
            ``frame`` contains finite joint positions.
        Postconditions:
            Returns a new frame with minimum joint z equal to ``clearance``.
        """

        return SkeletonFrame(
            joint_positions=self.transform_points(frame.joint_positions).astype(np.float32),
            root_quaternion_wxyz=frame.root_quaternion_wxyz,
            frame_index=frame.frame_index,
            label=f"{frame.label}_ground",
        )


class CoordinateTransformFactory:
    """Create coordinate transforms from CLI names.

    Responsibilities:
        Keep transform selection outside visualization and data-source classes.
    Preconditions:
        Names come from argparse choices.
    Postconditions:
        Returns a coordinate transform strategy object.
    """

    def create(self, name: str) -> CoordinateTransform:
        """Create a transform strategy.

        Preconditions:
            ``name`` is one of the supported transform names.
        Postconditions:
            Returns a concrete ``CoordinateTransform`` implementation.
        """

        if name == "identity":
            return IdentityCoordinateTransform()
        if name == "smpl-to-mujoco":
            return SmplToMujocoCoordinateTransform()
        raise ValueError(f"Unsupported coordinate transform: {name}")


class MujocoSkeletonSceneBuilder:
    """Build a static MuJoCo scene from one SMPL skeleton frame.

    Responsibilities:
        Encode SMPL joints directly as MuJoCo world-space spheres and bones,
        with robot-industry axes ``+X front``, ``+Y left``, ``+Z up``.
    Preconditions:
        Input frames contain finite joint positions in the coordinates to be
        inspected.  No coordinate conversion is applied by this builder.
    Postconditions:
        Returns valid MJCF XML that can be loaded by MuJoCo.
    """

    def __init__(self, config: Optional[MujocoSceneConfig] = None) -> None:
        """Create a MuJoCo scene builder.

        Preconditions:
            ``config`` is ``None`` or contains positive visual sizes.
        Postconditions:
            The builder can convert one skeleton frame into MJCF.
        """

        self._config = config or MujocoSceneConfig()
        if self._config.joint_radius <= 0:
            raise ValueError("joint_radius must be positive")
        if self._config.bone_radius <= 0:
            raise ValueError("bone_radius must be positive")
        if self._config.axis_radius <= 0:
            raise ValueError("axis_radius must be positive")
        if self._config.axis_length <= 0:
            raise ValueError("axis_length must be positive")

    def build(self, frame: SkeletonFrame) -> str:
        """Build a MuJoCo XML string for one skeleton frame.

        Preconditions:
            ``frame.joint_positions`` has shape ``(J, 3)`` and finite values.
        Postconditions:
            Returns a complete MJCF document with fixed skeleton geoms.
        """

        joints = np.asarray(frame.joint_positions, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[1] != 3:
            raise ValueError("joint_positions must have shape (J, 3)")
        if not np.isfinite(joints).all():
            raise ValueError("joint_positions contains non-finite values")

        lines = [
            '<mujoco model="smpl_skeleton_debug">',
            '  <compiler angle="radian" coordinate="local"/>',
            '  <option timestep="0.01" gravity="0 0 -9.81"/>',
            "  <visual>",
            '    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>',
            "  </visual>",
            "  <asset>",
            '    <material name="mat_joint" rgba="0.1 0.35 0.95 1"/>',
            '    <material name="mat_root" rgba="1 0.1 0.1 1"/>',
            '    <material name="mat_bone" rgba="0.05 0.05 0.05 1"/>',
            '    <material name="mat_x_front" rgba="1 0 0 1"/>',
            '    <material name="mat_y_left" rgba="0 0.75 0 1"/>',
            '    <material name="mat_z_up" rgba="0.1 0.25 1 1"/>',
            '    <material name="mat_floor" rgba="0.75 0.75 0.75 0.35"/>',
            "  </asset>",
            "  <worldbody>",
            '    <light name="key_light" pos="2 -3 4" dir="-2 3 -4" directional="true"/>',
            '    <camera name="debug_camera" pos="3 -4 2.2" xyaxes="0.8 0.6 0 -0.28 0.37 0.88"/>',
        ]
        if self._config.add_floor:
            lines.append('    <geom name="floor" type="plane" size="3 3 0.02" material="mat_floor"/>')
        lines.extend(self._build_axis_geoms())
        lines.extend(self._build_joint_bodies(joints))
        lines.extend(self._build_bone_geoms(joints))
        lines.extend(
            [
                "  </worldbody>",
                "</mujoco>",
            ]
        )
        return "\n".join(lines) + "\n"

    def _build_axis_geoms(self) -> list[str]:
        """Build world-axis geoms.

        Preconditions:
            Scene configuration is valid.
        Postconditions:
            Returns three capsules named by robot coordinate convention.
        """

        length = self._config.axis_length
        radius = self._config.axis_radius
        return [
            f'    <geom name="axis_x_front" type="capsule" fromto="0 0 0 {length:.6f} 0 0" '
            f'size="{radius:.6f}" material="mat_x_front"/>',
            f'    <geom name="axis_y_left" type="capsule" fromto="0 0 0 0 {length:.6f} 0" '
            f'size="{radius:.6f}" material="mat_y_left"/>',
            f'    <geom name="axis_z_up" type="capsule" fromto="0 0 0 0 0 {length:.6f}" '
            f'size="{radius:.6f}" material="mat_z_up"/>',
        ]

    def _build_joint_bodies(self, joints: np.ndarray) -> list[str]:
        """Build fixed joint marker bodies.

        Preconditions:
            ``joints`` has shape ``(J, 3)``.
        Postconditions:
            Returns one body with one sphere geom per joint.
        """

        lines = []
        for index, position in enumerate(joints):
            material = "mat_root" if index == 0 else "mat_joint"
            radius = self._config.joint_radius * (1.35 if index == 0 else 1.0)
            lines.extend(
                [
                    f'    <body name="smpl_joint_{index:02d}" pos="{self._format_vec(position)}">',
                    f'      <geom name="smpl_joint_{index:02d}_geom" type="sphere" size="{radius:.6f}" material="{material}"/>',
                    "    </body>",
                ]
            )
        return lines

    def _build_bone_geoms(self, joints: np.ndarray) -> list[str]:
        """Build capsule geoms between SMPL parent-child joints.

        Preconditions:
            ``joints`` has shape ``(J, 3)``.
        Postconditions:
            Returns one world-space capsule per valid SMPL edge.
        """

        lines = []
        for edge_index, (start, end) in enumerate(SMPL_JOINT_EDGES):
            if start >= len(joints) or end >= len(joints):
                continue
            start_pos = joints[start]
            end_pos = joints[end]
            if np.linalg.norm(end_pos - start_pos) < 1e-7:
                continue
            lines.append(
                f'    <geom name="smpl_bone_{edge_index:02d}_{start:02d}_{end:02d}" '
                f'type="capsule" fromto="{self._format_vec(start_pos)} {self._format_vec(end_pos)}" '
                f'size="{self._config.bone_radius:.6f}" material="mat_bone"/>'
            )
        return lines

    @staticmethod
    def _format_vec(vector: np.ndarray) -> str:
        """Format a vector for MJCF attributes.

        Preconditions:
            ``vector`` has three finite components.
        Postconditions:
            Returns a MuJoCo-compatible space-separated vector string.
        """

        return " ".join(f"{float(value):.6f}" for value in vector)


class MujocoSkeletonViewer:
    """Launch a MuJoCo viewer for a generated skeleton scene.

    Responsibilities:
        Keep optional MuJoCo GUI handling separate from XML generation.
    Preconditions:
        ``mujoco`` is importable and a GUI/display is available.
    Postconditions:
        The scene is displayed until the viewer window closes.
    """

    def show(self, xml: str) -> None:
        """Open the MuJoCo viewer for an MJCF string.

        Preconditions:
            ``xml`` is valid MJCF.
        Postconditions:
            Blocks while the passive viewer window is running.
        """

        import time
        import mujoco
        import mujoco.viewer

        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "debug_camera")
            if camera_id >= 0:
                viewer.cam.fixedcamid = camera_id
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)


class SvgSkeletonRenderer:
    """Render skeleton frames to a dependency-free SVG preview.

    Responsibilities:
        Provide a no-GUI fallback renderer when Matplotlib is unavailable.
    Preconditions:
        ``output_image`` is either ``None`` or points to a writable location.
    Postconditions:
        Writes an SVG file containing the first selected frame from each source.
    """

    _WIDTH_PER_PANEL = 520
    _HEIGHT = 520
    _MARGIN = 52

    def __init__(self, config: VisualizationConfig, fallback_reason: Optional[str] = None) -> None:
        """Create an SVG renderer.

        Preconditions:
            ``config`` is valid.
        Postconditions:
            The renderer can save a static SVG skeleton preview.
        """

        self._config = config
        self._fallback_reason = fallback_reason

    def render(self, frame_sets: Sequence[Sequence[SkeletonFrame]]) -> None:
        """Write an SVG preview.

        Preconditions:
            ``frame_sets`` contains at least one non-empty frame sequence.
        Postconditions:
            An SVG preview file is written to disk.
        """

        if not frame_sets or any(not frames for frames in frame_sets):
            raise ValueError("At least one non-empty frame set is required")
        output_path = self._resolve_output_path()
        first_frames = [frames[0] for frames in frame_sets]
        all_points = np.concatenate([frame.joint_positions for frame in first_frames], axis=0)
        projected = self._project_points(all_points)
        min_xy = projected.min(axis=0)
        max_xy = projected.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)
        width = self._WIDTH_PER_PANEL * len(first_frames)
        height = self._HEIGHT

        cursor = 0
        body_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]
        if self._fallback_reason is not None:
            body_lines.append(
                f'<text x="16" y="24" font-family="monospace" font-size="12" fill="#555">'
                f'Fallback SVG renderer: {escape(self._fallback_reason[:180])}</text>'
            )
        for panel_index, frame in enumerate(first_frames):
            count = len(frame.joint_positions)
            panel_points = projected[cursor : cursor + count]
            cursor += count
            screen_points = self._to_screen(panel_points, min_xy, span, panel_index)
            body_lines.extend(self._draw_frame_svg(frame, screen_points))
        body_lines.append("</svg>")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(body_lines) + "\n", encoding="utf-8")
        print(f"Saved skeleton preview to: {output_path}")

    def _resolve_output_path(self) -> Path:
        """Choose a safe SVG output path.

        Preconditions:
            None.
        Postconditions:
            Returns a path with a ``.svg`` suffix.
        """

        if self._config.output_image is None:
            return Path("skeleton_preview.svg")
        if self._config.output_image.suffix.lower() == ".svg":
            return self._config.output_image
        svg_path = self._config.output_image.with_suffix(".svg")
        print(f"Matplotlib image export unavailable; writing SVG fallback instead: {svg_path}")
        return svg_path

    @staticmethod
    def _project_points(points: np.ndarray) -> np.ndarray:
        """Project 3D points to a stable isometric 2D plane.

        Preconditions:
            ``points`` has shape ``(N, 3)``.
        Postconditions:
            Returns ``(N, 2)`` projected coordinates.
        """

        projection = np.array([[1.0, -0.45], [0.0, -0.28], [0.0, 1.0]], dtype=np.float32)
        return np.asarray(points, dtype=np.float32) @ projection

    def _to_screen(
        self,
        points: np.ndarray,
        min_xy: np.ndarray,
        span: np.ndarray,
        panel_index: int,
    ) -> np.ndarray:
        """Map projected points into one SVG panel.

        Preconditions:
            ``points`` has shape ``(N, 2)`` and ``span`` is non-zero.
        Postconditions:
            Returns SVG pixel coordinates for drawing.
        """

        panel_origin_x = panel_index * self._WIDTH_PER_PANEL
        drawable_width = self._WIDTH_PER_PANEL - 2 * self._MARGIN
        drawable_height = self._HEIGHT - 2 * self._MARGIN
        normalized = (points - min_xy) / span
        scale = min(drawable_width, drawable_height)
        x = panel_origin_x + self._MARGIN + normalized[:, 0] * scale
        y = self._HEIGHT - self._MARGIN - normalized[:, 1] * scale
        return np.stack([x, y], axis=1)

    def _draw_frame_svg(self, frame: SkeletonFrame, points: np.ndarray) -> list[str]:
        """Draw one skeleton frame as SVG primitives.

        Preconditions:
            ``points`` contains one projected point per skeleton joint.
        Postconditions:
            Returns SVG lines and circles for the frame.
        """

        elements = [
            f'<text x="{points[:, 0].min():.1f}" y="48" font-family="monospace" font-size="14" fill="#222">'
            f'{escape(frame.label)} frame {frame.frame_index}</text>'
        ]
        for start, end in SMPL_JOINT_EDGES:
            if start < len(points) and end < len(points):
                x1, y1 = points[start]
                x2, y2 = points[end]
                elements.append(
                    f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                    'stroke="#111" stroke-width="2" stroke-linecap="round"/>'
                )
        for index, (x_pos, y_pos) in enumerate(points):
            fill = "#d62728" if index == 0 else "#1f77b4"
            radius = 4.8 if index == 0 else 3.2
            elements.append(f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" r="{radius}" fill="{fill}"/>')
        return elements


class SkeletonRenderer(AbstractContextManager):
    """Render skeleton frames with Matplotlib and SVG fallback.

    Responsibilities:
        Own Matplotlib figures when available and delegate to SVG when the
        plotting stack cannot be imported.
    Preconditions:
        Frames have compatible joint counts.
    Postconditions:
        ``render`` either displays an interactive figure or writes a preview.
    """

    def __init__(self, config: VisualizationConfig) -> None:
        """Create a renderer.

        Preconditions:
            ``config`` is valid.
        Postconditions:
            The renderer is ready to create a figure lazily.
        """

        self._config = config
        self._figure = None
        self._plt = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the owned figure when leaving a context.

        Preconditions:
            May be called with or without a created figure.
        Postconditions:
            Matplotlib resources are released when they were allocated.
        """

        if self._figure is not None and self._plt is not None:
            self._plt.close(self._figure)

    def render(self, frame_sets: Sequence[Sequence[SkeletonFrame]]) -> None:
        """Render one frame from each source.

        Preconditions:
            ``frame_sets`` contains at least one non-empty frame sequence.
        Postconditions:
            Writes ``output_image`` when configured; otherwise opens a window.
        """

        if not frame_sets or any(not frames for frames in frame_sets):
            raise ValueError("At least one non-empty frame set is required")
        if self._config.output_image is not None and self._config.output_image.suffix.lower() == ".svg":
            SvgSkeletonRenderer(self._config).render(frame_sets)
            return
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                import matplotlib.pyplot as plt
        except Exception as error:
            if self._config.output_image is None:
                raise RuntimeError("Matplotlib is required for interactive display; pass --output-image preview.svg") from error
            SvgSkeletonRenderer(self._config, fallback_reason=str(error)).render(frame_sets)
            return

        self._plt = plt
        self._figure = plt.figure(figsize=(6 * len(frame_sets), 6))
        axes = []
        for index, frames in enumerate(frame_sets, start=1):
            axis = self._figure.add_subplot(1, len(frame_sets), index, projection="3d")
            self._draw_frame(axis, frames[0])
            axes.append(axis)
        self._set_equal_limits(axes, frame_sets)
        self._figure.tight_layout()
        if self._config.output_image is not None:
            self._figure.savefig(self._config.output_image)
            print(f"Saved skeleton preview to: {self._config.output_image}")
        else:
            plt.show()

    def _draw_frame(self, axis, frame: SkeletonFrame) -> None:
        """Draw one skeleton frame on a Matplotlib 3D axis.

        Preconditions:
            ``axis`` is a 3D Matplotlib axis.
        Postconditions:
            Skeleton lines, joints, and optional root axes are added to ``axis``.
        """

        joints = frame.joint_positions
        for start, end in SMPL_JOINT_EDGES:
            if start < len(joints) and end < len(joints):
                segment = joints[[start, end]]
                axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="black", linewidth=1.5)
        axis.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color="tab:blue", s=18)
        axis.scatter(joints[0:1, 0], joints[0:1, 1], joints[0:1, 2], color="tab:red", s=36)
        if self._config.show_root_axes and frame.root_quaternion_wxyz is not None:
            self._draw_root_axes(axis, joints[0], frame.root_quaternion_wxyz)
        axis.set_title(f"{frame.label} frame {frame.frame_index}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")

    def _draw_root_axes(self, axis, origin: np.ndarray, quaternion: np.ndarray) -> None:
        """Draw root coordinate axes.

        Preconditions:
            ``origin`` has shape ``(3,)`` and ``quaternion`` has shape ``(4,)``.
        Postconditions:
            Colored x/y/z root axes are drawn from ``origin``.
        """

        rotation_matrix = RotationAdapter.quat_wxyz_to_matrix(quaternion)
        colors = ("tab:red", "tab:green", "tab:blue")
        labels = ("root x", "root y", "root z")
        for axis_idx, color, label in zip(range(3), colors, labels):
            vector = rotation_matrix[:, axis_idx] * self._config.axis_length
            axis.quiver(
                origin[0],
                origin[1],
                origin[2],
                vector[0],
                vector[1],
                vector[2],
                color=color,
                label=label,
            )

    @staticmethod
    def _set_equal_limits(axes: Sequence[object], frame_sets: Sequence[Sequence[SkeletonFrame]]) -> None:
        """Set equal-ish limits on all axes.

        Preconditions:
            ``axes`` and ``frame_sets`` have matching lengths.
        Postconditions:
            Skeletons are not visually distorted by unequal axis scales.
        """

        all_joints = np.concatenate(
            [np.concatenate([frame.joint_positions for frame in frames[:1]], axis=0) for frames in frame_sets],
            axis=0,
        )
        center = all_joints.mean(axis=0)
        span = float(np.ptp(all_joints, axis=0).max())
        radius = max(span * 0.55, 0.5)
        for axis in axes:
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)


class SkeletonSourceFactory:
    """Build skeleton sources from CLI arguments.

    Responsibilities:
        Keep argument interpretation outside source and renderer classes.
    Preconditions:
        At least one source path is provided.
    Postconditions:
        Returns configured source objects without loading data.
    """

    def create_sources(self, args: argparse.Namespace, config: VisualizationConfig) -> list[SkeletonSource]:
        """Create skeleton sources.

        Preconditions:
            ``args`` is produced by ``build_argument_parser``.
        Postconditions:
            Returns one source per requested input.
        """

        sources: list[SkeletonSource] = []
        if args.reference is not None:
            sources.append(ReferenceSkeletonSource(args.reference, config=config))
        if args.npz is not None:
            if args.smpl_model is None:
                raise ValueError("--smpl-model is required when using --npz")
            sources.append(
                NpzSmplSkeletonSource(
                    args.npz,
                    SmplModelRunner(args.smpl_model, batch_size=args.batch_size),
                    config=config,
                )
            )
        if not sources:
            raise ValueError("Provide at least one of --reference or --npz")
        return sources


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Preconditions:
        None.
    Postconditions:
        Returns a parser for the visualization CLI.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help="Single deploy reference motion folder.")
    parser.add_argument("--npz", type=Path, help="Raw SMPL npz file.")
    parser.add_argument("--smpl-model", type=Path, help="SMPL model path used with --npz.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--no-root-axes", action="store_true")
    parser.add_argument("--axis-length", type=float, default=0.25)
    parser.add_argument("--output-image", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--coordinate-transform",
        choices=("identity", "smpl-to-mujoco"),
        default="identity",
        help="Coordinate transform applied only to MuJoCo scene export/viewer.",
    )
    parser.add_argument("--mujoco-xml", type=Path, help="Write the first selected skeleton frame to this MJCF XML file.")
    parser.add_argument("--mujoco-viewer", action="store_true", help="Open the first selected skeleton frame in MuJoCo.")
    parser.add_argument(
        "--mujoco-ground-align",
        action="store_true",
        help="Lift MuJoCo preview skeleton so its lowest joint is on the floor.",
    )
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=0.0,
        help="Extra z clearance used with --mujoco-ground-align.",
    )
    return parser


def main() -> None:
    """Run the visualization command-line interface.

    Preconditions:
        Command-line arguments point to readable inputs.
    Postconditions:
        Prints source summaries and renders or saves a preview image.
    """

    args = build_argument_parser().parse_args()
    config = VisualizationConfig(
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        stride=args.stride,
        show_root_axes=not args.no_root_axes,
        axis_length=args.axis_length,
        output_image=args.output_image,
    )
    sources = SkeletonSourceFactory().create_sources(args, config)
    frame_sets = [source.load() for source in sources]
    validator = SkeletonFrameValidator()
    for frames in frame_sets:
        print(f"{frames[0].label}: {validator.summarize(frames)}")
    if args.mujoco_xml is not None or args.mujoco_viewer:
        transform = CoordinateTransformFactory().create(args.coordinate_transform)
        if args.mujoco_ground_align:
            transform = GroundAlignmentTransform(transform, clearance=args.ground_clearance)
        mujoco_frame = transform.transform_frame(frame_sets[0][0])
        print(f"{mujoco_frame.label}: {validator.summarize([mujoco_frame])}")
        xml = MujocoSkeletonSceneBuilder().build(mujoco_frame)
        if args.mujoco_xml is not None:
            args.mujoco_xml.parent.mkdir(parents=True, exist_ok=True)
            args.mujoco_xml.write_text(xml, encoding="utf-8")
            print(f"Saved MuJoCo skeleton scene to: {args.mujoco_xml}")
        if args.mujoco_viewer:
            MujocoSkeletonViewer().show(xml)
        return
    if not args.summary_only:
        with SkeletonRenderer(config) as renderer:
            renderer.render(frame_sets)


if __name__ == "__main__":
    main()
