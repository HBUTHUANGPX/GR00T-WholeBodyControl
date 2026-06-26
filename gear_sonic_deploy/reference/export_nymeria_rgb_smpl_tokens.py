#!/usr/bin/env python3
"""Export RGB-frame-aligned Nymeria SMPL tokens for Sonic FSQ conditioning.

Each output token corresponds to one RGB frame, but each token's SMPL encoder
observation is still built from a strict 50 Hz, 10-frame future window:

    rgb_time + [0, 20 ms, 40 ms, ..., 180 ms]

The source SMPL motion is interpolated onto those window timestamps, and samples
outside the SMPL range are clamped to the first/last SMPL frame.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from convert_smpl_npz_to_reference import PicoSmplReferenceProcessor, SmplNpzData
from encode_motion_tokens import (
    ENCODER_INPUT_DIM,
    ENCODER_OUTPUT_DIM,
    OBS_LAYOUT,
    calc_heading_quat,
    create_encoder_session,
    encode_tokens,
    normalize_quat,
    quat_conjugate,
    quat_mul,
    quat_to_rot6d_first_two_cols,
)
from export_smpl_tokens import (
    DEFAULT_BASE_QUAT,
    DEFAULT_ENCODER_CHUNK_SIZE,
    DEFAULT_SMPL_BATCH_SIZE,
    ENCODE_MODE_SMPL,
    SMPL_FUTURE_FRAMES,
    WRIST_JOINT_INDICES,
)


FUTURE_STEP_NS = 20_000_000
DEFAULT_OUTPUT_SUFFIX = "_rgb_motion_token"
DEFAULT_ENCODER_MODEL = Path(__file__).resolve().parents[1] / "policy" / "release" / "model_encoder.onnx"
NYMERIA_TIME_DOMAIN = "time_code"
NYMERIA_RGB_TIME_ZERO_SOURCE = "recording_head/rgb/frame_0"


@dataclass(frozen=True)
class RgbAlignedWindowPlan:
    """RGB frame anchors and strict-50Hz SMPL sampling windows."""

    rgb_positions: np.ndarray
    rgb_frame_indices: np.ndarray
    rgb_relative_timestamps_ns: np.ndarray
    sample_relative_timestamps_ns: np.ndarray
    unclamped_sample_relative_timestamps_ns: np.ndarray
    clamped_sample_mask: np.ndarray


@dataclass(frozen=True)
class InterpolatedSmplMotion:
    """SMPL arrays interpolated to RGB-anchored future windows."""

    global_orient: np.ndarray
    body_pose: np.ndarray
    transl: np.ndarray
    betas: np.ndarray
    left_indices: np.ndarray
    right_indices: np.ndarray
    alpha: np.ndarray


@dataclass(frozen=True)
class NymeriaSmplPayload:
    """Loaded Nymeria SMPL data and timestamp metadata."""

    path: Path
    global_orient: np.ndarray
    body_pose: np.ndarray
    transl: np.ndarray
    betas: np.ndarray
    relative_timestamps_ns: np.ndarray
    frame_indices: np.ndarray | None
    time_zero_ns: int
    time_zero_source: str
    time_domain: str
    time_alignment_version: int | None


@dataclass(frozen=True)
class RgbTimestampPayload:
    """Loaded RGB timestamp sidecar metadata."""

    path: Path
    relative_timestamps_ns: np.ndarray
    frame_indices: np.ndarray
    time_zero_ns: int
    time_zero_source: str
    time_domain: str
    time_alignment_version: int | None


@dataclass(frozen=True)
class WindowedSmplReference:
    """Pico-processed SMPL reference arrays reshaped as [RGB frame, future]."""

    smpl_joints_local: np.ndarray
    body_quat_w: np.ndarray
    joint_pos: np.ndarray


@dataclass(frozen=True)
class ExportResult:
    output_path: Path
    frames: int


@dataclass(frozen=True)
class SequenceInput:
    """Resolved converted-sequence paths for one export job."""

    sequence_dir: Path
    smpl_npz: Path
    rgb_timestamps_npz: Path
    output_path: Path


def build_rgb_aligned_window_plan(
    smpl_relative_timestamps_ns: np.ndarray,
    rgb_relative_timestamps_ns: np.ndarray,
    rgb_frame_indices: np.ndarray | None = None,
    *,
    future_frames: int = SMPL_FUTURE_FRAMES,
    future_step_ns: int = FUTURE_STEP_NS,
    max_rgb_frames: int | None = None,
) -> RgbAlignedWindowPlan:
    """Build RGB anchors and 50Hz future-window sample times.

    The first RGB anchor is the RGB frame immediately before the first SMPL
    frame. If there is no such RGB frame, the first RGB frame is used.
    """

    smpl_times = _require_strictly_increasing_int64(smpl_relative_timestamps_ns, "smpl_relative_timestamps_ns")
    rgb_times = _require_increasing_int64(rgb_relative_timestamps_ns, "rgb_relative_timestamps_ns", min_size=1)
    if future_frames <= 0:
        raise ValueError("future_frames must be positive")
    if future_step_ns <= 0:
        raise ValueError("future_step_ns must be positive")
    if rgb_frame_indices is None:
        rgb_frame_indices = np.arange(len(rgb_times), dtype=np.int32)
    else:
        rgb_frame_indices = np.asarray(rgb_frame_indices, dtype=np.int32).reshape(-1)
        if rgb_frame_indices.shape[0] != rgb_times.shape[0]:
            raise ValueError("rgb_frame_indices must match rgb_relative_timestamps_ns length")

    start = int(np.searchsorted(rgb_times, smpl_times[0], side="right")) - 1
    start = max(0, start)
    rgb_positions = np.arange(start, len(rgb_times), dtype=np.int64)
    if max_rgb_frames is not None:
        if max_rgb_frames <= 0:
            raise ValueError("max_rgb_frames must be positive when provided")
        rgb_positions = rgb_positions[: int(max_rgb_frames)]
    if rgb_positions.size == 0:
        raise ValueError("No RGB frames selected")

    anchors = rgb_times[rgb_positions]
    offsets = np.arange(future_frames, dtype=np.int64) * np.int64(future_step_ns)
    raw_sample_times = anchors[:, None] + offsets[None, :]
    sample_times = np.clip(raw_sample_times, smpl_times[0], smpl_times[-1]).astype(np.int64, copy=False)
    return RgbAlignedWindowPlan(
        rgb_positions=rgb_positions,
        rgb_frame_indices=rgb_frame_indices[rgb_positions],
        rgb_relative_timestamps_ns=anchors,
        sample_relative_timestamps_ns=sample_times,
        unclamped_sample_relative_timestamps_ns=raw_sample_times,
        clamped_sample_mask=sample_times != raw_sample_times,
    )


def interpolate_smpl_motion(
    smpl_relative_timestamps_ns: np.ndarray,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    betas: np.ndarray,
    sample_relative_timestamps_ns: np.ndarray,
) -> InterpolatedSmplMotion:
    """Interpolate SMPL pose arrays to arbitrary timestamp windows."""

    smpl_times = _require_strictly_increasing_int64(smpl_relative_timestamps_ns, "smpl_relative_timestamps_ns")
    sample_times = np.asarray(sample_relative_timestamps_ns, dtype=np.int64)
    flat_times = sample_times.reshape(-1)
    if flat_times.size == 0:
        raise ValueError("sample_relative_timestamps_ns cannot be empty")
    if flat_times.min() < smpl_times[0] or flat_times.max() > smpl_times[-1]:
        raise ValueError("sample timestamps must be clamped to the SMPL timestamp range before interpolation")

    global_orient = _require_shape(np.asarray(global_orient, dtype=np.float32), (len(smpl_times), 3), "global_orient")
    body_pose = _require_shape(np.asarray(body_pose, dtype=np.float32), (len(smpl_times), 69), "body_pose")
    transl = _require_shape(np.asarray(transl, dtype=np.float32), (len(smpl_times), 3), "transl")
    betas = _normalize_betas(np.asarray(betas, dtype=np.float32), len(smpl_times))

    right = np.searchsorted(smpl_times, flat_times, side="right")
    right = np.clip(right, 1, len(smpl_times) - 1)
    left = right - 1
    denom = (smpl_times[right] - smpl_times[left]).astype(np.float64)
    alpha_flat = ((flat_times - smpl_times[left]).astype(np.float64) / denom).astype(np.float32)
    sample_shape = sample_times.shape
    alpha = alpha_flat.reshape(sample_shape)

    sampled_transl = _lerp(transl, left, right, alpha_flat).reshape(*sample_shape, 3)
    sampled_betas = _lerp(betas, left, right, alpha_flat).reshape(*sample_shape, betas.shape[1])

    pose = np.concatenate([global_orient[:, None, :], body_pose.reshape(len(smpl_times), 23, 3)], axis=1)
    q0 = rotvec_to_quat_wxyz(pose[left])
    q1 = rotvec_to_quat_wxyz(pose[right])
    sampled_pose = quat_wxyz_to_rotvec(quat_slerp_wxyz(q0, q1, alpha_flat[:, None]))
    sampled_pose = sampled_pose.reshape(*sample_shape, 24, 3).astype(np.float32, copy=False)

    return InterpolatedSmplMotion(
        global_orient=sampled_pose[..., 0, :],
        body_pose=sampled_pose[..., 1:, :].reshape(*sample_shape, 69),
        transl=sampled_transl.astype(np.float32, copy=False),
        betas=sampled_betas.astype(np.float32, copy=False),
        left_indices=left.reshape(sample_shape),
        right_indices=right.reshape(sample_shape),
        alpha=alpha,
    )


def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors to wxyz quaternions."""

    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half = 0.5 * angle
    small = angle < 1e-12
    scale = np.empty_like(angle)
    np.divide(np.sin(half), angle, out=scale, where=~small)
    scale[small] = 0.5
    quat = np.concatenate([np.cos(half), rotvec * scale], axis=-1)
    return normalize_quat(quat)


def quat_wxyz_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions to shortest-path axis-angle vectors."""

    quat = normalize_quat(quat)
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)
    vector = quat[..., 1:]
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.clip(quat[..., :1], -1.0, 1.0))
    scale = np.empty_like(norm)
    np.divide(angle, norm, out=scale, where=norm >= 1e-12)
    scale[norm < 1e-12] = 2.0
    return (vector * scale).astype(np.float32, copy=False)


def quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation for wxyz quaternions."""

    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    alpha = np.asarray(alpha, dtype=np.float64)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    linear = dot > 0.9995

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha[..., None] if alpha.ndim == q0.ndim - 1 else theta_0 * alpha
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / np.maximum(sin_theta_0, 1e-12)
    s1 = sin_theta / np.maximum(sin_theta_0, 1e-12)
    out = s0 * q0 + s1 * q1
    linear_out = normalize_quat((1.0 - alpha[..., None]) * q0 + alpha[..., None] * q1)
    return normalize_quat(np.where(linear, linear_out, out))


def load_nymeria_smpl_payload(path: Path) -> NymeriaSmplPayload:
    """Load a dataset_converter Nymeria SMPL NPZ."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"global_orient", "body_pose", "transl", "betas", "time_zero_ns"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing required SMPL fields: {missing}")
        time_zero_ns = _scalar_int(data["time_zero_ns"])
        if "relative_timestamps_ns" in data.files:
            relative_timestamps_ns = np.asarray(data["relative_timestamps_ns"], dtype=np.int64).reshape(-1)
        elif "timestamps_ns" in data.files:
            relative_timestamps_ns = np.asarray(data["timestamps_ns"], dtype=np.int64).reshape(-1) - time_zero_ns
        else:
            raise ValueError(f"{path} is missing relative_timestamps_ns or timestamps_ns")
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int32).reshape(-1) if "frame_indices" in data.files else None
        return NymeriaSmplPayload(
            path=path,
            global_orient=np.asarray(data["global_orient"], dtype=np.float32),
            body_pose=np.asarray(data["body_pose"], dtype=np.float32),
            transl=np.asarray(data["transl"], dtype=np.float32),
            betas=np.asarray(data["betas"], dtype=np.float32),
            relative_timestamps_ns=relative_timestamps_ns,
            frame_indices=frame_indices,
            time_zero_ns=time_zero_ns,
            time_zero_source=_scalar_str(data["time_zero_source"]) if "time_zero_source" in data.files else "",
            time_domain=_scalar_str(data["time_domain"]) if "time_domain" in data.files else "",
            time_alignment_version=_scalar_int(data["nymeria_time_alignment_version"])
            if "nymeria_time_alignment_version" in data.files
            else None,
        )


def load_rgb_timestamp_payload(path: Path) -> RgbTimestampPayload:
    """Load dataset_converter RGB timestamp sidecar."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"rgb_relative_timestamps_ns", "time_zero_ns"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing required RGB timestamp fields: {missing}")
        relative_timestamps_ns = np.asarray(data["rgb_relative_timestamps_ns"], dtype=np.int64).reshape(-1)
        frame_indices = (
            np.asarray(data["rgb_frame_indices"], dtype=np.int32).reshape(-1)
            if "rgb_frame_indices" in data.files
            else np.arange(len(relative_timestamps_ns), dtype=np.int32)
        )
        return RgbTimestampPayload(
            path=path,
            relative_timestamps_ns=relative_timestamps_ns,
            frame_indices=frame_indices,
            time_zero_ns=_scalar_int(data["time_zero_ns"]),
            time_zero_source=_scalar_str(data["time_zero_source"]) if "time_zero_source" in data.files else "",
            time_domain=_scalar_str(data["time_domain"]) if "time_domain" in data.files else "",
            time_alignment_version=_scalar_int(data["nymeria_time_alignment_version"])
            if "nymeria_time_alignment_version" in data.files
            else None,
        )


def validate_shared_time_axis(smpl: NymeriaSmplPayload, rgb: RgbTimestampPayload) -> None:
    """Ensure SMPL and RGB sidecars are using the same RGB-zero time axis."""

    if smpl.time_zero_ns != rgb.time_zero_ns:
        raise ValueError(f"time_zero_ns mismatch: smpl={smpl.time_zero_ns}, rgb={rgb.time_zero_ns}")
    for label, payload in (("smpl", smpl), ("rgb", rgb)):
        if payload.time_domain and payload.time_domain != NYMERIA_TIME_DOMAIN:
            raise ValueError(f"{label} time_domain must be {NYMERIA_TIME_DOMAIN!r}, got {payload.time_domain!r}")
        if payload.time_zero_source and payload.time_zero_source != NYMERIA_RGB_TIME_ZERO_SOURCE:
            raise ValueError(
                f"{label} time_zero_source must be {NYMERIA_RGB_TIME_ZERO_SOURCE!r}, got {payload.time_zero_source!r}"
            )
    if (
        smpl.time_alignment_version is not None
        and rgb.time_alignment_version is not None
        and smpl.time_alignment_version != rgb.time_alignment_version
    ):
        raise ValueError(
            "nymeria_time_alignment_version mismatch: "
            f"smpl={smpl.time_alignment_version}, rgb={rgb.time_alignment_version}"
        )


def process_interpolated_windows(
    sampled: InterpolatedSmplMotion,
    *,
    name: str,
    batch_size: int = DEFAULT_SMPL_BATCH_SIZE,
) -> WindowedSmplReference:
    """Run Pico-style SMPL processing and reshape back into RGB/future windows."""

    rows, future_frames = sampled.global_orient.shape[:2]
    flat_frames = rows * future_frames
    raw = SmplNpzData(
        name=name,
        global_orient=sampled.global_orient.reshape(flat_frames, 3).astype(np.float32, copy=False),
        body_pose=sampled.body_pose.reshape(flat_frames, 69).astype(np.float32, copy=False),
        transl=sampled.transl.reshape(flat_frames, 3).astype(np.float32, copy=False),
        betas=sampled.betas.reshape(flat_frames, -1).astype(np.float32, copy=False),
        source_frame_indices=np.arange(flat_frames, dtype=np.int64),
        source_fps=50.0,
        target_fps=50.0,
    )
    reference = PicoSmplReferenceProcessor(batch_size=batch_size).process(raw)
    return WindowedSmplReference(
        smpl_joints_local=reference.smpl_joints_local.reshape(rows, future_frames, 24, 3),
        body_quat_w=reference.body_quat_w.reshape(rows, future_frames, 4),
        joint_pos=reference.joint_pos.reshape(rows, future_frames, 29),
    )


def build_rgb_aligned_encoder_observations(
    reference: WindowedSmplReference,
    *,
    base_quat: np.ndarray | None = None,
) -> np.ndarray:
    """Build one SMPL-mode encoder observation per RGB frame."""

    smpl_joints = np.asarray(reference.smpl_joints_local[:, :, :24, :], dtype=np.float32)
    body_quat = normalize_quat(np.asarray(reference.body_quat_w, dtype=np.float64))
    joint_pos = np.asarray(reference.joint_pos, dtype=np.float32)
    rows, future_frames = smpl_joints.shape[:2]
    if future_frames != SMPL_FUTURE_FRAMES:
        raise ValueError(f"Expected {SMPL_FUTURE_FRAMES} future frames, got {future_frames}")

    obs = np.zeros((rows, ENCODER_INPUT_DIM), dtype=np.float32)
    mode_offset, _ = OBS_LAYOUT["encoder_mode_4"]
    obs[:, mode_offset] = ENCODE_MODE_SMPL

    base_quat = normalize_quat(
        np.asarray([1.0, 0.0, 0.0, 0.0] if base_quat is None else base_quat, dtype=np.float64)
    )
    apply_delta_heading = quat_mul(
        calc_heading_quat(base_quat),
        calc_heading_quat(body_quat[0, 0], inverse=True),
    )

    smpl_offset, _ = OBS_LAYOUT["smpl_joints_10frame_step1"]
    ori_offset, _ = OBS_LAYOUT["smpl_anchor_orientation_10frame_step1"]
    wrist_offset, _ = OBS_LAYOUT["motion_joint_positions_wrists_10frame_step1"]

    obs[:, smpl_offset : smpl_offset + future_frames * 24 * 3] = smpl_joints.reshape(rows, -1)
    new_ref_quat = quat_mul(apply_delta_heading, body_quat)
    relative_quat = quat_mul(quat_conjugate(base_quat), new_ref_quat)
    obs[:, ori_offset : ori_offset + future_frames * 6] = quat_to_rot6d_first_two_cols(relative_quat).reshape(rows, -1)
    obs[:, wrist_offset : wrist_offset + future_frames * len(WRIST_JOINT_INDICES)] = (
        joint_pos[:, :, WRIST_JOINT_INDICES].reshape(rows, -1)
    )
    return obs


def make_progress(total: int, desc: str, unit: str, *, enabled: bool = True):
    """Create an optional tqdm progress bar without making tqdm a hard dependency."""

    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def close_progress(progress) -> None:
    if progress is None:
        return
    close = getattr(progress, "close", None)
    if close is not None:
        close()


def iter_frame_chunks(total: int, chunk_size: int, progress=None):
    """Yield half-open frame ranges and update progress by frame count."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield start, end
        if progress is not None:
            progress.update(end - start)


def encode_observations_chunked(encoder_session, obs: np.ndarray, chunk_size: int, progress=None) -> np.ndarray:
    """Run the fixed-batch ONNX encoder over observations."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks = []
    for start, end in iter_frame_chunks(len(obs), chunk_size, progress=progress):
        chunks.append(encode_tokens(encoder_session, obs[start:end]))
    return np.concatenate(chunks, axis=0)


def save_rgb_aligned_token_npz(
    path: Path,
    *,
    tokens: np.ndarray,
    smpl: NymeriaSmplPayload,
    rgb: RgbTimestampPayload,
    plan: RgbAlignedWindowPlan,
    sampled: InterpolatedSmplMotion,
    encoder_model: Path,
) -> None:
    """Save token_state and enough metadata to map rows back to RGB/SMPL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        token_state=tokens.astype(np.float32, copy=False),
        motion_length=np.asarray(tokens.shape[0], dtype=np.int64),
        frame_indices=np.arange(tokens.shape[0], dtype=np.int64),
        rgb_positions=plan.rgb_positions.astype(np.int64, copy=False),
        rgb_frame_indices=plan.rgb_frame_indices.astype(np.int32, copy=False),
        rgb_relative_timestamps_ns=plan.rgb_relative_timestamps_ns.astype(np.int64, copy=False),
        smpl_sample_relative_timestamps_ns=plan.sample_relative_timestamps_ns.astype(np.int64, copy=False),
        smpl_unclamped_sample_relative_timestamps_ns=plan.unclamped_sample_relative_timestamps_ns.astype(
            np.int64, copy=False
        ),
        smpl_sample_clamped_mask=plan.clamped_sample_mask.astype(np.bool_, copy=False),
        source_smpl_left_indices=sampled.left_indices.astype(np.int64, copy=False),
        source_smpl_right_indices=sampled.right_indices.astype(np.int64, copy=False),
        source_smpl_interp_alpha=sampled.alpha.astype(np.float32, copy=False),
        source_smpl_npz=str(smpl.path.resolve()),
        source_rgb_timestamps_npz=str(rgb.path.resolve()),
        encoder_model=str(encoder_model.resolve()),
        encoder_input_dim=np.asarray(ENCODER_INPUT_DIM, dtype=np.int64),
        encoder_output_dim=np.asarray(ENCODER_OUTPUT_DIM, dtype=np.int64),
        encoder_mode=np.asarray(int(ENCODE_MODE_SMPL), dtype=np.int64),
        future_frames=np.asarray(SMPL_FUTURE_FRAMES, dtype=np.int64),
        future_step_ns=np.asarray(FUTURE_STEP_NS, dtype=np.int64),
        future_step_hz=np.asarray(50.0, dtype=np.float32),
        sampling_policy="rgb_anchor_50hz_future_window_slerp_clamp",
        time_zero_ns=np.asarray(smpl.time_zero_ns, dtype=np.int64),
        time_zero_source=smpl.time_zero_source or NYMERIA_RGB_TIME_ZERO_SOURCE,
        time_domain=smpl.time_domain or NYMERIA_TIME_DOMAIN,
        nymeria_time_alignment_version=np.asarray(-1 if smpl.time_alignment_version is None else smpl.time_alignment_version),
        feature_schema={
            "filled_observations": [
                "encoder_mode_4",
                "smpl_joints_10frame_step1",
                "smpl_anchor_orientation_10frame_step1",
                "motion_joint_positions_wrists_10frame_step1",
            ],
            "obs_layout": OBS_LAYOUT,
            "unused_observations": "zero_filled",
        },
    )


def default_output_path(sequence_dir: Path) -> Path:
    return Path(sequence_dir) / "token" / "token.npz"


def is_converted_sequence_dir(path: Path) -> bool:
    path = Path(path)
    return (path / "smpl" / "nymeria_smpl.npz").is_file() and (path / "head_video" / "timestamps.npz").is_file()


def discover_sequence_inputs(root: Path, *, output_name: str = "token.npz") -> list[SequenceInput]:
    """Discover converted Nymeria sequence directories under root."""

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Input root not found: {root}")
    candidates = [root] if is_converted_sequence_dir(root) else []
    candidates.extend(path.parent.parent for path in root.rglob("smpl/nymeria_smpl.npz") if is_converted_sequence_dir(path.parent.parent))
    sequence_dirs = sorted(set(candidates))
    return [
        SequenceInput(
            sequence_dir=sequence_dir,
            smpl_npz=sequence_dir / "smpl" / "nymeria_smpl.npz",
            rgb_timestamps_npz=sequence_dir / "head_video" / "timestamps.npz",
            output_path=sequence_dir / "token" / output_name,
        )
        for sequence_dir in sequence_dirs
    ]


def export_sequence_rgb_aligned_tokens(
    *,
    smpl_npz: Path,
    rgb_timestamps_npz: Path,
    output_path: Path,
    encoder_model: Path,
    overwrite: bool = False,
    max_rgb_frames: int | None = None,
    smpl_batch_size: int = DEFAULT_SMPL_BATCH_SIZE,
    encoder_chunk_size: int = DEFAULT_ENCODER_CHUNK_SIZE,
    progress_enabled: bool = True,
    progress_label: str | None = None,
) -> ExportResult:
    """Export one RGB-frame-aligned Nymeria SMPL token file."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}")
    smpl = load_nymeria_smpl_payload(smpl_npz)
    rgb = load_rgb_timestamp_payload(rgb_timestamps_npz)
    validate_shared_time_axis(smpl, rgb)
    plan = build_rgb_aligned_window_plan(
        smpl.relative_timestamps_ns,
        rgb.relative_timestamps_ns,
        rgb.frame_indices,
        max_rgb_frames=max_rgb_frames,
    )
    sampled = interpolate_smpl_motion(
        smpl.relative_timestamps_ns,
        smpl.global_orient,
        smpl.body_pose,
        smpl.transl,
        smpl.betas,
        plan.sample_relative_timestamps_ns,
    )
    reference = process_interpolated_windows(
        sampled,
        name=smpl_npz.stem,
        batch_size=smpl_batch_size,
    )
    obs = build_rgb_aligned_encoder_observations(reference, base_quat=DEFAULT_BASE_QUAT)
    session = create_encoder_session(encoder_model)
    frame_progress = make_progress(
        len(obs),
        desc=f"{progress_label or output_path.parent.parent.name} RGB frames",
        unit="frame",
        enabled=progress_enabled,
    )
    try:
        tokens = encode_observations_chunked(session, obs, encoder_chunk_size, progress=frame_progress)
    finally:
        close_progress(frame_progress)
    if tokens.shape != (len(plan.rgb_frame_indices), ENCODER_OUTPUT_DIM):
        raise RuntimeError(f"Unexpected token shape {tokens.shape}")
    save_rgb_aligned_token_npz(
        output_path,
        tokens=tokens,
        smpl=smpl,
        rgb=rgb,
        plan=plan,
        sampled=sampled,
        encoder_model=encoder_model,
    )
    return ExportResult(output_path=output_path, frames=tokens.shape[0])


def resolve_input_paths(sequence_dir: Path | None, smpl_npz: Path | None, rgb_timestamps_npz: Path | None) -> tuple[Path, Path]:
    if sequence_dir is not None:
        smpl_npz = smpl_npz or sequence_dir / "smpl" / "nymeria_smpl.npz"
        rgb_timestamps_npz = rgb_timestamps_npz or sequence_dir / "head_video" / "timestamps.npz"
    if smpl_npz is None or rgb_timestamps_npz is None:
        raise ValueError("Provide either sequence_dir or both --smpl-npz and --rgb-timestamps")
    return Path(smpl_npz), Path(rgb_timestamps_npz)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export one Sonic FSQ token per Nymeria RGB frame using strict 50Hz SMPL future windows.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Converted Nymeria sequence directory, or a root containing many sequence directories.",
    )
    parser.add_argument("--smpl-npz", type=Path, help="Explicit Nymeria SMPL NPZ path.")
    parser.add_argument("--rgb-timestamps", type=Path, help="Explicit head_video/timestamps.npz path.")
    parser.add_argument("--encoder-model", type=Path, default=DEFAULT_ENCODER_MODEL, help="Path to model_encoder.onnx.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output token NPZ path. Only valid with an explicit single sequence or explicit --smpl-npz/--rgb-timestamps.",
    )
    parser.add_argument("--output-name", default="token.npz", help="Token filename inside each sequence token/ folder.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    parser.add_argument("--max-rgb-frames", type=int, help="Optional debug limit for selected RGB anchors.")
    parser.add_argument("--smpl-batch-size", type=int, default=DEFAULT_SMPL_BATCH_SIZE)
    parser.add_argument("--encoder-chunk-size", type=int, default=DEFAULT_ENCODER_CHUNK_SIZE)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    encoder_model = args.encoder_model
    if not encoder_model.exists():
        parser.error(f"encoder model not found: {encoder_model}")
    if args.output is not None and args.input is not None and not is_converted_sequence_dir(args.input):
        parser.error("--output can only be used with a single converted sequence directory")

    if args.smpl_npz is not None or args.rgb_timestamps is not None:
        try:
            smpl_npz, rgb_timestamps_npz = resolve_input_paths(None, args.smpl_npz, args.rgb_timestamps)
        except ValueError as exc:
            parser.error(str(exc))
        jobs = [
            SequenceInput(
                sequence_dir=smpl_npz.parents[1],
                smpl_npz=smpl_npz,
                rgb_timestamps_npz=rgb_timestamps_npz,
                output_path=args.output or smpl_npz.parents[1] / "token" / args.output_name,
            )
        ]
    else:
        if args.input is None:
            parser.error("Provide an input root/sequence directory or explicit --smpl-npz and --rgb-timestamps")
        jobs = discover_sequence_inputs(args.input, output_name=args.output_name)
        if args.output is not None and len(jobs) == 1:
            jobs = [
                SequenceInput(
                    sequence_dir=jobs[0].sequence_dir,
                    smpl_npz=jobs[0].smpl_npz,
                    rgb_timestamps_npz=jobs[0].rgb_timestamps_npz,
                    output_path=args.output,
                )
            ]
    if not jobs:
        parser.error(f"No converted Nymeria sequences found under: {args.input}")

    print("Nymeria RGB-Aligned SMPL Token Exporter")
    print("=======================================")
    print(f"Encoder: {encoder_model.resolve()}")
    print(f"Sequences: {len(jobs)}")
    print(f"Future window: {SMPL_FUTURE_FRAMES} frames @ 50Hz, clamp_to_smpl_bounds")
    converted = 0
    failed = 0
    skipped = 0
    sequence_progress = make_progress(len(jobs), desc="Sequences", unit="sequence", enabled=not args.no_progress)
    try:
        for index, job in enumerate(jobs, start=1):
            print(f"\n[{index}/{len(jobs)}] {job.sequence_dir}")
            print(f"  SMPL: {job.smpl_npz}")
            print(f"  RGB timestamps: {job.rgb_timestamps_npz}")
            print(f"  Output: {job.output_path}")
            try:
                result = export_sequence_rgb_aligned_tokens(
                    smpl_npz=job.smpl_npz,
                    rgb_timestamps_npz=job.rgb_timestamps_npz,
                    output_path=job.output_path,
                    encoder_model=encoder_model,
                    overwrite=args.overwrite,
                    max_rgb_frames=args.max_rgb_frames,
                    smpl_batch_size=args.smpl_batch_size,
                    encoder_chunk_size=args.encoder_chunk_size,
                    progress_enabled=not args.no_progress,
                    progress_label=job.sequence_dir.name,
                )
            except FileExistsError as exc:
                print(f"  Skipped: {exc}")
                skipped += 1
            except Exception as exc:
                print(f"  Failed: {exc}")
                failed += 1
            else:
                print(f"  Saved token_state: {result.frames} RGB frames x {ENCODER_OUTPUT_DIM} dims")
                converted += 1
            finally:
                if sequence_progress is not None:
                    sequence_progress.update(1)
    finally:
        close_progress(sequence_progress)
    print(f"\nBatch complete: converted={converted}, skipped={skipped}, failed={failed}, total={len(jobs)}")


def _require_strictly_increasing_int64(value: np.ndarray, name: str) -> np.ndarray:
    return _require_increasing_int64(value, name, min_size=2)


def _require_increasing_int64(value: np.ndarray, name: str, *, min_size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64).reshape(-1)
    if array.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} timestamp(s)")
    if array.size > 1 and not np.all(np.diff(array) > 0):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def _require_shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalize_betas(betas: np.ndarray, frames: int) -> np.ndarray:
    if betas.ndim == 1:
        betas = np.broadcast_to(betas[None, :], (frames, betas.shape[0]))
    elif betas.ndim == 2 and betas.shape[0] == 1:
        betas = np.broadcast_to(betas, (frames, betas.shape[1]))
    elif betas.ndim != 2 or betas.shape[0] != frames:
        raise ValueError(f"betas has shape {betas.shape}, expected ({frames}, B), (1, B), or (B,)")
    if not np.isfinite(betas).all():
        raise ValueError("betas contains non-finite values")
    return np.asarray(betas, dtype=np.float32)


def _lerp(array: np.ndarray, left: np.ndarray, right: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    values = array[left] * (1.0 - alpha[:, None]) + array[right] * alpha[:, None]
    return values.astype(np.float32, copy=False)


def _scalar_int(value: np.ndarray) -> int:
    return int(np.asarray(value).reshape(()))


def _scalar_str(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(()))


if __name__ == "__main__":
    main()
