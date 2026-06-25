#!/usr/bin/env python3
"""Batch-export offline SMPL trajectory NPZ files to equal-length token NPZ files.

Usage:
    # Convert one SMPL trajectory. Output is saved next to the source file as:
    # /path/to/smpl_data_motion_token.npz
    python3 gear_sonic_deploy/reference/export_smpl_tokens.py /path/to/smpl_data.npz

    # Recursively scan a dataset directory with many nested SMPL trajectory NPZs.
    # Each token file is saved in the same directory as its source trajectory.
    python3 gear_sonic_deploy/reference/export_smpl_tokens.py /path/to/dataset_root

    # Recompute existing token files.
    python3 gear_sonic_deploy/reference/export_smpl_tokens.py /path/to/dataset_root --overwrite

    # Quick smoke test on a few frames.
    python3 gear_sonic_deploy/reference/export_smpl_tokens.py /path/to/smpl_data.npz --max-frames 100

Input contract:
    A source NPZ is treated as an SMPL trajectory only when it contains
    global_orient, body_pose, and transl. Other NPZ files, such as SMPL model
    files or previously exported token files, are ignored during directory scans.

Output contract:
    For a source /a/b/name.npz, the default output is /a/b/name_motion_token.npz.
    The output stores token_state with shape [selected_frames, 64].
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from convert_smpl_npz_to_reference import (
    ConversionConfig,
    PicoSmplReferenceProcessor,
    SmplNpzDataSource,
    SmplReferenceData,
)
from encode_motion_tokens import (
    ENCODER_INPUT_DIM,
    ENCODER_OUTPUT_DIM,
    OBS_LAYOUT,
    calc_heading_quat,
    create_encoder_session,
    encode_tokens,
    future_indices,
    normalize_quat,
    quat_conjugate,
    quat_mul,
    quat_to_rot6d_first_two_cols,
)


DEFAULT_INPUT_NPZ = Path("/home/jerry_huang/HPX_Loco/GR00T-WholeBodyControl/smpl_data.npz")
DEFAULT_ENCODER_MODEL = Path("gear_sonic_deploy/policy/release/model_encoder.onnx")
DEFAULT_OUTPUT_SUFFIX = "_motion_token"
DEFAULT_SOURCE_FPS = 50.0
DEFAULT_TARGET_FPS = 50.0
DEFAULT_SMPL_BATCH_SIZE = 4096
DEFAULT_ENCODER_CHUNK_SIZE = 2048
DEFAULT_BASE_QUAT = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
ENCODE_MODE_SMPL = 2.0
SMPL_FUTURE_FRAMES = 10
SMPL_FUTURE_STEP = 1
WRIST_JOINT_INDICES = np.asarray([23, 24, 25, 26, 27, 28], dtype=np.int64)
SMPL_REQUIRED_NPZ_KEYS = frozenset({"global_orient", "body_pose", "transl"})


@dataclass(frozen=True)
class ExportJobResult:
    """Outcome for one source NPZ in batch mode."""

    source: Path
    output: Path
    status: str
    message: str = ""
    frames: int = 0


def build_smpl_encoder_observations(
    reference: SmplReferenceData,
    base_quat: np.ndarray | None = None,
    frame_start: int = 0,
    frame_end: int | None = None,
) -> np.ndarray:
    """Build encoder observations for SMPL mode with one row per reference frame."""

    _validate_reference(reference)
    smpl_joints = reference.smpl_joints_local[:, :24, :].astype(np.float32, copy=False)
    body_quat = normalize_quat(reference.body_quat_w.astype(np.float64, copy=False))
    joint_pos = reference.joint_pos.astype(np.float32, copy=False)
    num_frames = smpl_joints.shape[0]
    frame_end = num_frames if frame_end is None else frame_end
    if frame_start < 0 or frame_end < frame_start or frame_end > num_frames:
        raise ValueError(f"Invalid frame slice [{frame_start}, {frame_end}) for {num_frames} frames")
    output_frames = frame_end - frame_start

    obs = np.zeros((output_frames, ENCODER_INPUT_DIM), dtype=np.float32)
    mode_offset, _ = OBS_LAYOUT["encoder_mode_4"]
    obs[:, mode_offset] = ENCODE_MODE_SMPL

    base_quat = normalize_quat(
        np.asarray([1.0, 0.0, 0.0, 0.0] if base_quat is None else base_quat, dtype=np.float64)
    )
    apply_delta_heading = quat_mul(
        calc_heading_quat(base_quat),
        calc_heading_quat(body_quat[0], inverse=True),
    )

    smpl_offset, _ = OBS_LAYOUT["smpl_joints_10frame_step1"]
    ori_offset, _ = OBS_LAYOUT["smpl_anchor_orientation_10frame_step1"]
    wrist_offset, _ = OBS_LAYOUT["motion_joint_positions_wrists_10frame_step1"]

    for row, frame in enumerate(range(frame_start, frame_end)):
        idx = future_indices(num_frames, frame, SMPL_FUTURE_FRAMES, SMPL_FUTURE_STEP)

        obs[row, smpl_offset : smpl_offset + SMPL_FUTURE_FRAMES * 24 * 3] = smpl_joints[idx].reshape(-1)

        new_ref_quat = quat_mul(apply_delta_heading, body_quat[idx])
        relative_quat = quat_mul(quat_conjugate(base_quat), new_ref_quat)
        obs[row, ori_offset : ori_offset + SMPL_FUTURE_FRAMES * 6] = (
            quat_to_rot6d_first_two_cols(relative_quat).reshape(-1).astype(np.float32)
        )

        wrist_values = joint_pos[idx][:, WRIST_JOINT_INDICES]
        obs[row, wrist_offset : wrist_offset + SMPL_FUTURE_FRAMES * len(WRIST_JOINT_INDICES)] = wrist_values.reshape(-1)

    return obs


def default_token_output_path(npz_path: Path, suffix: str = DEFAULT_OUTPUT_SUFFIX) -> Path:
    """Return the default token path next to the source SMPL trajectory npz."""

    return npz_path.with_name(f"{npz_path.stem}{suffix}.npz")


def is_smpl_trajectory_npz(npz_path: Path) -> bool:
    """Return True when an NPZ contains the required raw SMPL trajectory keys."""

    try:
        with np.load(npz_path) as data:
            return SMPL_REQUIRED_NPZ_KEYS.issubset(set(data.files))
    except Exception:
        return False


def discover_smpl_npz_files(input_path: Path, recursive: bool = True, pattern: str = "*.npz") -> list[Path]:
    """Discover SMPL trajectory NPZ files from one file or a directory tree."""

    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path] if is_smpl_trajectory_npz(input_path) else []
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    candidates = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
    return sorted(path for path in candidates if path.is_file() and is_smpl_trajectory_npz(path))


def encode_smpl_tokens_chunked(
    encoder_session,
    reference: SmplReferenceData,
    base_quat: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Encode SMPL observations in chunks while preserving one token per frame."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    token_chunks = []
    frames = len(reference.smpl_joints_local)
    for frame_start in range(0, frames, chunk_size):
        frame_end = min(frame_start + chunk_size, frames)
        obs = build_smpl_encoder_observations(
            reference,
            base_quat=base_quat,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        token_chunks.append(encode_tokens(encoder_session, obs))
    tokens = np.concatenate(token_chunks, axis=0)
    if tokens.shape[0] != frames:
        raise RuntimeError(f"Token length mismatch: tokens={tokens.shape[0]} frames={frames}")
    return tokens


def convert_smpl_npz_to_reference(
    npz_path: Path,
    config: ConversionConfig,
    batch_size: int,
) -> SmplReferenceData:
    """Load an SMPL npz and process it into deploy SMPL-reference arrays."""

    raw = SmplNpzDataSource(npz_path, config).load()
    return PicoSmplReferenceProcessor(batch_size=batch_size).process(raw)


def export_one_smpl_npz(
    npz_path: Path,
    output_path: Path,
    encoder_session,
    encoder_model: Path,
    config: ConversionConfig,
    overwrite: bool = False,
) -> ExportJobResult:
    """Convert one SMPL NPZ to tokens."""

    if output_path.exists() and not overwrite:
        return ExportJobResult(npz_path, output_path, "skipped", "output exists")

    reference = convert_smpl_npz_to_reference(npz_path, config, batch_size=DEFAULT_SMPL_BATCH_SIZE)
    tokens = encode_smpl_tokens_chunked(
        encoder_session,
        reference,
        base_quat=DEFAULT_BASE_QUAT,
        chunk_size=DEFAULT_ENCODER_CHUNK_SIZE,
    )
    save_token_npz(output_path, tokens, reference, npz_path, encoder_model)
    return ExportJobResult(npz_path, output_path, "converted", frames=tokens.shape[0])


def export_smpl_token_batch(
    npz_files: list[Path],
    encoder_session,
    encoder_model: Path,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> list[ExportJobResult]:
    """Run token export for a list of discovered SMPL NPZ files."""

    results: list[ExportJobResult] = []
    for index, npz_path in enumerate(npz_files, start=1):
        output_path = default_token_output_path(npz_path)
        config = ConversionConfig(
            max_frames=max_frames,
            source_fps=DEFAULT_SOURCE_FPS,
            target_fps=DEFAULT_TARGET_FPS,
        )
        print(f"\n[{index}/{len(npz_files)}] {npz_path}")
        print(f"  Output: {output_path}")
        try:
            result = export_one_smpl_npz(
                npz_path=npz_path,
                output_path=output_path,
                encoder_session=encoder_session,
                encoder_model=encoder_model,
                config=config,
                overwrite=overwrite,
            )
        except Exception as exc:
            result = ExportJobResult(npz_path, output_path, "failed", str(exc))
        if result.status == "converted":
            print(f"  Saved token_state: {result.frames} frames x {ENCODER_OUTPUT_DIM} dims")
        elif result.status == "skipped":
            print(f"  Skipped: {result.message}")
        else:
            print(f"  Failed: {result.message}")
        results.append(result)
    return results


def save_token_npz(
    path: Path,
    tokens: np.ndarray,
    reference: SmplReferenceData,
    npz_path: Path,
    encoder_model: Path,
) -> None:
    """Save tokens and metadata in the same token_state layout used by deploy."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token_state": tokens.astype(np.float32, copy=False),
        "frame_indices": np.arange(tokens.shape[0], dtype=np.int64),
        "motion_length": np.asarray(tokens.shape[0], dtype=np.int64),
        "source_npz": str(npz_path.resolve()),
        "encoder_model": str(encoder_model.resolve()),
        "encoder_input_dim": np.asarray(ENCODER_INPUT_DIM, dtype=np.int64),
        "encoder_output_dim": np.asarray(ENCODER_OUTPUT_DIM, dtype=np.int64),
        "encoder_mode": np.asarray(int(ENCODE_MODE_SMPL), dtype=np.int64),
        "future_frames": np.asarray(SMPL_FUTURE_FRAMES, dtype=np.int64),
        "future_step": np.asarray(SMPL_FUTURE_STEP, dtype=np.int64),
        "window_policy": "clamp_to_clip",
        "feature_schema": {
            "filled_observations": [
                "encoder_mode_4",
                "smpl_joints_10frame_step1",
                "smpl_anchor_orientation_10frame_step1",
                "motion_joint_positions_wrists_10frame_step1",
            ],
            "obs_layout": OBS_LAYOUT,
            "unused_observations": "zero_filled",
        },
    }
    if reference.source_frame_indices is not None:
        payload["source_frame_indices"] = reference.source_frame_indices.astype(np.int64, copy=False)
    if reference.source_fps is not None:
        payload["source_fps"] = np.asarray(reference.source_fps, dtype=np.float64)
    if reference.target_fps is not None:
        payload["target_fps"] = np.asarray(reference.target_fps, dtype=np.float64)
    np.savez_compressed(path, **payload)


def _validate_reference(reference: SmplReferenceData) -> None:
    frames = len(reference.smpl_joints_local)
    expected = {
        "smpl_joints_local": (frames, 24, 3),
        "body_quat_w": (frames, 4),
        "joint_pos": (frames, 29),
    }
    actual = {
        "smpl_joints_local": reference.smpl_joints_local.shape,
        "body_quat_w": reference.body_quat_w.shape,
        "joint_pos": reference.joint_pos.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} has shape {actual[name]}, expected {shape}")
    for name, array in (
        ("smpl_joints_local", reference.smpl_joints_local),
        ("body_quat_w", reference.body_quat_w),
        ("joint_pos", reference.joint_pos),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-convert SMPL trajectory NPZ files to equal-length SMPL-mode encoder tokens.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_NPZ,
        help="SMPL trajectory NPZ or a directory to scan recursively.",
    )
    parser.add_argument(
        "--encoder-model",
        type=Path,
        default=DEFAULT_ENCODER_MODEL,
        help="Path to model_encoder.onnx.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_motion_token.npz outputs instead of skipping them.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug limit. Omit to preserve every frame.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = args.input
    encoder_model = args.encoder_model

    if not input_path.exists():
        parser.error(f"Input path not found: {input_path}")
    if not encoder_model.exists():
        parser.error(f"Encoder model not found: {encoder_model}")

    npz_files = discover_smpl_npz_files(input_path)
    if not npz_files:
        parser.error(f"No SMPL trajectory NPZ files found under: {input_path}")

    print("SMPL Motion Token Exporter")
    print("==========================")
    print(f"Input: {input_path.resolve()}")
    print(f"Encoder: {encoder_model.resolve()}")
    print(f"Motions: {len(npz_files)}")
    print(f"Sampling: source_fps={DEFAULT_SOURCE_FPS}, target_fps={DEFAULT_TARGET_FPS}, max_frames={args.max_frames}")

    session = create_encoder_session(encoder_model)
    results = export_smpl_token_batch(
        npz_files,
        session,
        encoder_model,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )
    converted = sum(result.status == "converted" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    failed = sum(result.status == "failed" for result in results)
    print(f"\nBatch complete: converted={converted}, skipped={skipped}, failed={failed}, total={len(results)}")


if __name__ == "__main__":
    main()
