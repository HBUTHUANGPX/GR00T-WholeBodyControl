#!/usr/bin/env python3
"""Convert offline SMPL npz data into a deploy reference motion folder.

The generated reference follows the same SMPL encoder contract used by the
Pico streaming path: SMPL joints are canonicalized into root-local coordinates,
SMPL root quaternions are converted to the deploy convention, and robot wrist
joint placeholders are present for ``motion_joint_positions_wrists_*``
observations.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class SmplNpzData:
    """Raw SMPL arrays loaded from an npz file.

    Responsibilities:
        Carry validated SMPL parameter arrays from storage into processing.
    Preconditions:
        Arrays share the same frame dimension and contain finite numeric values.
    Postconditions:
        Processing classes can consume the data without re-reading the npz file.
    """

    name: str
    global_orient: np.ndarray
    body_pose: np.ndarray
    transl: np.ndarray
    betas: np.ndarray
    source_frame_indices: np.ndarray
    source_fps: float
    target_fps: float


@dataclass(frozen=True)
class SmplReferenceData:
    """Deploy-ready arrays for one offline SMPL reference motion.

    Responsibilities:
        Bundle exactly the signals needed by deploy SMPL encoder mode.
    Preconditions:
        All arrays share the same frame count; SMPL joints are ``(T, 24, 3)``.
    Postconditions:
        ``DeployReferenceCsvWriter`` can serialize the bundle directly.
    """

    name: str
    smpl_joints_local: np.ndarray
    smpl_pose: np.ndarray
    body_quat_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    source_frame_indices: Optional[np.ndarray] = None
    source_fps: Optional[float] = None
    target_fps: Optional[float] = None


@dataclass(frozen=True)
class ConversionConfig:
    """Configuration for offline SMPL conversion.

    Responsibilities:
        Store frame sampling and output naming choices.
    Preconditions:
        ``start_frame >= 0``; ``stride > 0``; FPS values are positive; and
        ``max_frames`` is positive or ``None``.
    Postconditions:
        Data source and processor use one consistent time sampling policy.
    """

    start_frame: int = 0
    max_frames: Optional[int] = 1000
    stride: int = 1
    motion_name: Optional[str] = None
    source_fps: float = 240.0
    target_fps: float = 50.0


class SmplNpzDataSource:
    """Load raw SMPL parameters from an npz file.

    Responsibilities:
        Validate required npz keys and apply deterministic frame slicing.
    Preconditions:
        The npz contains ``global_orient``, ``body_pose``, and ``transl``.
    Postconditions:
        Returns finite arrays with matching frame counts.
    """

    def __init__(self, npz_path: Path, config: ConversionConfig) -> None:
        """Create an npz data source.

        Preconditions:
            ``npz_path`` points to an existing SMPL npz file.
        Postconditions:
            ``load`` can return sliced SMPL arrays.
        """

        if config.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if config.stride <= 0:
            raise ValueError("stride must be positive")
        if config.max_frames is not None and config.max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if config.source_fps <= 0.0:
            raise ValueError("source_fps must be positive")
        if config.target_fps <= 0.0:
            raise ValueError("target_fps must be positive")
        self._npz_path = Path(npz_path)
        self._config = config

    def load(self) -> SmplNpzData:
        """Load and validate SMPL arrays.

        Preconditions:
            The npz file exists and has supported array shapes.
        Postconditions:
            Returns sliced arrays with shape ``(T, *)``.
        """

        if not self._npz_path.exists():
            raise FileNotFoundError(f"SMPL npz not found: {self._npz_path}")
        with np.load(self._npz_path) as data:
            raw_global_orient = self._require(data, "global_orient")
            raw_body_pose = self._require(data, "body_pose")
            raw_transl = self._require(data, "transl")
            frame_indices = self._frame_indices(len(raw_global_orient))
            global_orient = raw_global_orient[frame_indices].astype(np.float32)
            body_pose = raw_body_pose[frame_indices].astype(np.float32)
            transl = raw_transl[frame_indices].astype(np.float32)
            if "betas" in data:
                raw_betas = np.asarray(data["betas"])
                if raw_betas.ndim > 1 and len(raw_betas) == len(raw_global_orient):
                    betas = raw_betas[frame_indices].astype(np.float32)
                else:
                    betas = raw_betas.astype(np.float32)
            else:
                betas = np.zeros((len(global_orient), 10), dtype=np.float32)

        if betas.ndim == 1:
            betas = np.tile(betas[None, :], (len(global_orient), 1))
        if len(betas) == 1 and len(global_orient) > 1:
            betas = np.tile(betas, (len(global_orient), 1))
        frame_count = len(global_orient)
        if frame_count == 0:
            raise ValueError("No frames selected from SMPL npz")
        for name, array in (("body_pose", body_pose), ("transl", transl), ("betas", betas)):
            if len(array) != frame_count:
                raise ValueError(f"{name} frame count does not match global_orient")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        if not np.isfinite(global_orient).all():
            raise ValueError("global_orient contains non-finite values")
        motion_name = self._config.motion_name or f"{self._npz_path.stem}_{frame_count}"
        return SmplNpzData(
            name=motion_name,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas,
            source_frame_indices=frame_indices,
            source_fps=self._config.source_fps,
            target_fps=self._config.target_fps,
        )

    def _frame_indices(self, source_frame_count: int) -> np.ndarray:
        """Compute source frame indexes for target-rate playback.

        Preconditions:
            ``source_frame_count`` is the number of raw frames in the npz.
        Postconditions:
            Returns monotonically nondecreasing raw frame indexes sampled by
            source/target time, capped to available frames.
        """

        available_count = max(0, source_frame_count - self._config.start_frame)
        if available_count == 0:
            return np.asarray([], dtype=np.int64)

        source_frame_step = self._config.source_fps / self._config.target_fps
        if self._config.max_frames is None:
            output_count = int(np.floor((available_count - 1) / source_frame_step)) + 1
        else:
            output_count = self._config.max_frames

        raw_offsets = np.floor(np.arange(output_count, dtype=np.float64) * source_frame_step + 0.5).astype(np.int64)
        raw_offsets = raw_offsets[raw_offsets < available_count]
        if self._config.stride > 1:
            raw_offsets = raw_offsets[:: self._config.stride]
        return raw_offsets + self._config.start_frame

    @staticmethod
    def _require(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
        """Read a required npz key.

        Preconditions:
            ``data`` is an open npz file.
        Postconditions:
            Returns the requested array or raises ``ValueError``.
        """

        if key not in data:
            raise ValueError(f"Input npz is missing required key '{key}'")
        return np.asarray(data[key])


class PicoSmplReferenceProcessor:
    """Process SMPL parameters like the Pico online path.

    Responsibilities:
        Convert root orientation to deploy convention, run the light-weight SMPL
        joint FK helper, and rotate joints into root-local coordinates.
    Preconditions:
        The repository Python package is importable and contains SMPL joint
        metadata at ``gear_sonic/data/human/human_joints_info.pkl``.
    Postconditions:
        Returns SMPL encoder-ready local joints and root quaternion history.
    """

    def __init__(self, batch_size: int = 4096) -> None:
        """Create a processor.

        Preconditions:
            ``batch_size`` is positive.
        Postconditions:
            ``process`` can run conversion in bounded memory chunks.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._batch_size = batch_size

    def process(self, data: SmplNpzData) -> SmplReferenceData:
        """Convert raw SMPL npz arrays to deploy-ready reference arrays.

        Preconditions:
            ``data.body_pose`` has at least 63 columns and root orientations
            are axis-angle vectors.
        Postconditions:
            Returns arrays suitable for ``MotionDataReader`` and SMPL encoder
            mode.
        """

        import torch
        from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
        from gear_sonic.trl.utils.torch_transform import (
            angle_axis_to_quaternion,
            compute_human_joints,
            quat_apply,
            quat_inv,
            quaternion_to_angle_axis,
        )

        if data.body_pose.shape[1] < 63:
            raise ValueError("body_pose must contain at least 63 columns")

        local_joint_chunks = []
        root_quat_chunks = []
        for start in range(0, len(data.global_orient), self._batch_size):
            end = min(start + self._batch_size, len(data.global_orient))
            global_orient = torch.as_tensor(data.global_orient[start:end], dtype=torch.float32)
            body_pose = torch.as_tensor(data.body_pose[start:end, :63], dtype=torch.float32)

            root_quat_y_up = angle_axis_to_quaternion(global_orient)
            root_quat_z_up = smpl_root_ytoz_up(root_quat_y_up)
            global_orient_z_up = quaternion_to_angle_axis(root_quat_z_up)

            joints = compute_human_joints(
                body_pose=body_pose,
                global_orient=global_orient_z_up,
            )

            root_quat_processed = remove_smpl_base_rot(root_quat_z_up, w_last=False)
            root_quat_inv = quat_inv(root_quat_processed).unsqueeze(1).repeat(1, joints.shape[1], 1)
            joints_local = quat_apply(root_quat_inv, joints)
            local_joint_chunks.append(joints_local.detach().cpu().numpy().astype(np.float32))
            root_quat_chunks.append(root_quat_processed.detach().cpu().numpy().astype(np.float32))

        smpl_joints_local = np.concatenate(local_joint_chunks, axis=0)
        body_quat_w = np.concatenate(root_quat_chunks, axis=0)
        smpl_pose = data.body_pose[:, :63].reshape(len(data.body_pose), 21, 3).astype(np.float32)
        joint_pos = np.zeros((len(data.body_pose), 29), dtype=np.float32)
        joint_vel = np.zeros_like(joint_pos)
        return SmplReferenceData(
            name=data.name,
            smpl_joints_local=smpl_joints_local,
            smpl_pose=smpl_pose,
            body_quat_w=body_quat_w,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            source_frame_indices=data.source_frame_indices,
            source_fps=data.source_fps,
            target_fps=data.target_fps,
        )


class DeployReferenceCsvWriter:
    """Write deploy reference CSV files for one motion.

    Responsibilities:
        Serialize SMPL local joints, root orientation, and robot joint
        placeholders in the layout consumed by ``MotionDataReader``.
    Preconditions:
        ``output_root`` is writable.
    Postconditions:
        A motion directory with CSV, metadata, and info files exists.
    """

    def __init__(self, output_root: Path) -> None:
        """Create a writer.

        Preconditions:
            The parent of ``output_root`` exists or can be created.
        Postconditions:
            ``write`` can create motion subdirectories.
        """

        self._output_root = Path(output_root)

    def write(self, data: SmplReferenceData) -> Path:
        """Write one deploy reference motion directory.

        Preconditions:
            All arrays in ``data`` share the same frame count.
        Postconditions:
            Returns the created motion directory path.
        """

        self._validate(data)
        motion_dir = self._output_root / data.name
        motion_dir.mkdir(parents=True, exist_ok=True)
        frames = len(data.smpl_joints_local)

        self._write_csv(motion_dir / "smpl_joint.csv", data.smpl_joints_local.reshape(frames, -1), "smpl_joint")
        self._write_csv(motion_dir / "smpl_pose.csv", data.smpl_pose.reshape(frames, -1), "smpl_pose")
        self._write_csv(motion_dir / "body_quat.csv", data.body_quat_w.reshape(frames, -1), "body")
        self._write_csv(motion_dir / "joint_pos.csv", data.joint_pos, "joint")
        self._write_csv(motion_dir / "joint_vel.csv", data.joint_vel, "joint")
        self._write_csv(motion_dir / "body_pos.csv", np.zeros((frames, 3), dtype=np.float32), "body_pos")
        self._write_csv(motion_dir / "body_lin_vel.csv", np.zeros((frames, 3), dtype=np.float32), "body_lin_vel")
        self._write_csv(motion_dir / "body_ang_vel.csv", np.zeros((frames, 3), dtype=np.float32), "body_ang_vel")
        self._write_metadata(motion_dir / "metadata.txt", data)
        self._write_info(motion_dir / "info.txt", data)
        return motion_dir

    @staticmethod
    def _validate(data: SmplReferenceData) -> None:
        """Validate output array shapes.

        Preconditions:
            ``data`` is constructed by a processor or test fixture.
        Postconditions:
            Raises a clear exception when deploy CSVs would be malformed.
        """

        frames = len(data.smpl_joints_local)
        expected = {
            "smpl_joints_local": (frames, 24, 3),
            "smpl_pose": (frames, 21, 3),
            "body_quat_w": (frames, 4),
            "joint_pos": (frames, 29),
            "joint_vel": (frames, 29),
        }
        actual = {
            "smpl_joints_local": data.smpl_joints_local.shape,
            "smpl_pose": data.smpl_pose.shape,
            "body_quat_w": data.body_quat_w.shape,
            "joint_pos": data.joint_pos.shape,
            "joint_vel": data.joint_vel.shape,
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} has shape {actual[name]}, expected {shape}")
        for name, array in (
            ("smpl_joints_local", data.smpl_joints_local),
            ("smpl_pose", data.smpl_pose),
            ("body_quat_w", data.body_quat_w),
            ("joint_pos", data.joint_pos),
            ("joint_vel", data.joint_vel),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")

    @staticmethod
    def _write_csv(path: Path, array: np.ndarray, prefix: str) -> None:
        """Write a 2D numeric CSV with a deploy-compatible header.

        Preconditions:
            ``array`` is two-dimensional.
        Postconditions:
            ``path`` contains one header row followed by numeric rows.
        """

        if array.ndim != 2:
            raise ValueError(f"CSV array must be 2D for {path}")
        headers = [f"{prefix}_{idx}" for idx in range(array.shape[1])]
        with path.open("w", encoding="utf-8") as handle:
            handle.write(",".join(headers) + "\n")
            np.savetxt(handle, array, delimiter=",", fmt="%.9f")

    @staticmethod
    def _write_metadata(path: Path, data: SmplReferenceData) -> None:
        """Write a lightweight metadata file.

        Preconditions:
            ``data`` has been validated.
        Postconditions:
            MotionDataReader can parse at least the root body index and humans
            can inspect the file.
        """

        frames = len(data.smpl_joints_local)
        content = (
            f"Metadata for: {data.name}\n"
            "==============================\n\n"
            "Body part indexes:\n"
            "[ 0]\n\n"
            f"Total timesteps: {frames}\n\n"
            "Data arrays summary:\n"
            f"  joint_pos: ({frames}, 29) (float32)\n"
            f"  joint_vel: ({frames}, 29) (float32)\n"
            f"  body_quat_w: ({frames}, 1, 4) (float32)\n"
            f"  smpl_joints: ({frames}, 24, 3) (float32)\n"
            f"  smpl_pose: ({frames}, 21, 3) (float32)\n"
        )
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _write_info(path: Path, data: SmplReferenceData) -> None:
        """Write a short human-readable summary.

        Preconditions:
            ``data`` has been validated.
        Postconditions:
            The motion directory includes a quick inspection summary.
        """

        frames = len(data.smpl_joints_local)
        root_abs = float(np.abs(data.smpl_joints_local[:, 0, :]).max())
        sampling_summary = ""
        if data.source_frame_indices is not None and len(data.source_frame_indices) > 0:
            sampling_summary = (
                f"source_fps: {data.source_fps}\n"
                f"target_fps: {data.target_fps}\n"
                f"source_frame_first: {int(data.source_frame_indices[0])}\n"
                f"source_frame_last: {int(data.source_frame_indices[-1])}\n"
            )
        content = (
            f"Motion Information: {data.name}\n"
            "==================================================\n\n"
            f"frames: {frames}\n"
            f"{sampling_summary}"
            "encoder_mode: smpl (2)\n"
            "smpl_joints: root-local, Pico-style y-up to z-up root processing\n"
            f"smpl_joint_root_max_abs: {root_abs:.9f}\n"
            "wrist_joint_placeholders: zeros at G1 wrist indices [23, 24, 25, 26, 27, 28]\n"
        )
        path.write_text(content, encoding="utf-8")


class SmplReferenceConversionApp:
    """Application service for offline SMPL reference conversion.

    Responsibilities:
        Coordinate loading, processing, and writing without embedding CLI logic
        in lower-level classes.
    Preconditions:
        Input npz exists and output root is writable.
    Postconditions:
        A deploy reference motion folder exists on success.
    """

    def __init__(
        self,
        source: SmplNpzDataSource,
        processor: PicoSmplReferenceProcessor,
        writer: DeployReferenceCsvWriter,
    ) -> None:
        """Create the conversion app.

        Preconditions:
            Dependencies implement the expected load/process/write behavior.
        Postconditions:
            ``run`` performs one complete conversion.
        """

        self._source = source
        self._processor = processor
        self._writer = writer

    def run(self) -> Path:
        """Run conversion end to end.

        Preconditions:
            Dependencies are correctly configured.
        Postconditions:
            Returns the output motion directory.
        """

        raw = self._source.load()
        reference = self._processor.process(raw)
        return self._writer.write(reference)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Preconditions:
        None.
    Postconditions:
        Returns a parser for offline SMPL conversion.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True, help="Input SMPL npz file.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("gear_sonic_deploy/reference/offline_smpl"),
        help="Directory that will contain the generated motion folder.",
    )
    parser.add_argument("--motion-name", type=str, help="Generated motion folder name.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=1000, help="Maximum output frames at target FPS.")
    parser.add_argument("--stride", type=int, default=1, help="Optional stride applied after FPS resampling.")
    parser.add_argument("--source-fps", type=float, default=240.0, help="Frame rate of the input SMPL npz.")
    parser.add_argument("--target-fps", type=float, default=50.0, help="Frame rate expected by deploy reference playback.")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser


def main() -> None:
    """Run the offline SMPL conversion CLI.

    Preconditions:
        Command-line arguments point to readable input and writable output.
    Postconditions:
        Prints the created motion directory path.
    """

    args = build_argument_parser().parse_args()
    config = ConversionConfig(
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        stride=args.stride,
        motion_name=args.motion_name,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
    )
    app = SmplReferenceConversionApp(
        source=SmplNpzDataSource(args.npz, config),
        processor=PicoSmplReferenceProcessor(batch_size=args.batch_size),
        writer=DeployReferenceCsvWriter(args.output_root),
    )
    motion_dir = app.run()
    print(f"Saved deploy SMPL reference motion to: {motion_dir}")


if __name__ == "__main__":
    main()
