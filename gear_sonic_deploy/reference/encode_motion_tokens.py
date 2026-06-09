#!/usr/bin/env python3
"""
Encode deployment reference motions into per-frame robot motion tokens.

This script builds the same 1762-D encoder input layout used by
gear_sonic_deploy/policy/release/model_encoder.onnx, fills only the G1 robot
motion encoder fields, pads future frames by clamping to the final frame, and
saves one 64-D token per source frame.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np


ENCODER_INPUT_DIM = 1762
ENCODER_OUTPUT_DIM = 64
FUTURE_FRAMES = 10
FUTURE_STEP = 5
ENCODE_MODE_G1 = 0.0

OBS_LAYOUT = {
    "encoder_mode_4": (0, 4),
    "motion_joint_positions_10frame_step5": (4, 290),
    "motion_joint_velocities_10frame_step5": (294, 290),
    "motion_root_z_position_10frame_step5": (584, 10),
    "motion_root_z_position": (594, 1),
    "motion_anchor_orientation": (595, 6),
    "motion_anchor_orientation_10frame_step5": (601, 60),
    "motion_joint_positions_lowerbody_10frame_step5": (661, 120),
    "motion_joint_velocities_lowerbody_10frame_step5": (781, 120),
    "vr_3point_local_target": (901, 9),
    "vr_3point_local_orn_target": (910, 12),
    "smpl_joints_10frame_step1": (922, 720),
    "smpl_anchor_orientation_10frame_step1": (1642, 60),
    "motion_joint_positions_wrists_10frame_step1": (1702, 60),
}


def load_csv_matrix(path):
    """Load a CSV file with one header row into a float32 matrix."""
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)


def discover_motion_dirs(input_path):
    """Return motion directories that contain the needed reference CSV files."""
    input_path = Path(input_path)
    required = {"joint_pos.csv", "joint_vel.csv", "body_quat.csv"}
    if input_path.is_dir() and required.issubset({p.name for p in input_path.iterdir()}):
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input must be a motion directory or parent directory: {input_path}")
    return sorted(
        child
        for child in input_path.iterdir()
        if child.is_dir() and required.issubset({p.name for p in child.iterdir()})
    )


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-12)


def quat_mul(a, b):
    """Quaternion multiply for wxyz quaternions."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_conjugate(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[..., 1:] *= -1.0
    return q


def quat_rotate(q, v):
    """Rotate vector v by wxyz quaternion q."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    q_w = q[..., :1]
    q_vec = q[..., 1:]
    return (
        v * (2.0 * q_w * q_w - 1.0)
        + np.cross(q_vec, v) * q_w * 2.0
        + q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    )


def calc_heading_quat(q, inverse=False):
    """Return yaw-only heading quaternion for wxyz quaternion q."""
    ref_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    rot_dir = quat_rotate(q, ref_dir)
    heading = np.arctan2(rot_dir[..., 1], rot_dir[..., 0])
    if inverse:
        heading = -heading
    half = heading * 0.5
    zeros = np.zeros_like(half)
    return normalize_quat(np.stack([np.cos(half), zeros, zeros, np.sin(half)], axis=-1))


def quat_to_rot6d_first_two_cols(q):
    """Convert wxyz quaternion to row-wise flattened first two matrix columns."""
    q = normalize_quat(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    return np.stack([r00, r01, r10, r11, r20, r21], axis=-1)


def future_indices(num_frames, start_frame, future_frames=FUTURE_FRAMES, step=FUTURE_STEP):
    idx = start_frame + np.arange(future_frames, dtype=np.int64) * step
    return np.minimum(idx, num_frames - 1)


def build_g1_encoder_observations(joint_pos, joint_vel, root_quat, base_quat):
    """Build the 1762-D encoder input matrix, filling only G1 mode fields."""
    num_frames = joint_pos.shape[0]
    obs = np.zeros((num_frames, ENCODER_INPUT_DIM), dtype=np.float32)

    offset, _ = OBS_LAYOUT["encoder_mode_4"]
    obs[:, offset] = ENCODE_MODE_G1

    base_quat = normalize_quat(np.asarray(base_quat, dtype=np.float64))
    init_ref_quat = root_quat[0].astype(np.float64)
    apply_delta_heading = quat_mul(
        calc_heading_quat(base_quat),
        calc_heading_quat(init_ref_quat, inverse=True),
    )

    pos_offset, _ = OBS_LAYOUT["motion_joint_positions_10frame_step5"]
    vel_offset, _ = OBS_LAYOUT["motion_joint_velocities_10frame_step5"]
    ori_offset, _ = OBS_LAYOUT["motion_anchor_orientation_10frame_step5"]

    for frame in range(num_frames):
        idx = future_indices(num_frames, frame)
        obs[frame, pos_offset : pos_offset + FUTURE_FRAMES * 29] = joint_pos[idx].reshape(-1)
        obs[frame, vel_offset : vel_offset + FUTURE_FRAMES * 29] = joint_vel[idx].reshape(-1)

        new_ref_quat = quat_mul(apply_delta_heading, root_quat[idx].astype(np.float64))
        relative_quat = quat_mul(quat_conjugate(base_quat), new_ref_quat)
        obs[frame, ori_offset : ori_offset + FUTURE_FRAMES * 6] = (
            quat_to_rot6d_first_two_cols(relative_quat).reshape(-1).astype(np.float32)
        )

    return obs


def create_encoder_session(encoder_model):
    """Create and validate an ONNX Runtime encoder session."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required. Use .venv_sim/bin/python or install onnxruntime."
        ) from exc

    session = ort.InferenceSession(str(encoder_model), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.name != "obs_dict":
        raise ValueError(f"Unexpected encoder input name: {input_meta.name}")
    if output_meta.name != "encoded_tokens":
        raise ValueError(f"Unexpected encoder output name: {output_meta.name}")
    return session


def encode_tokens(session, obs):
    """Run the ONNX encoder and return token_state with shape [T, 64]."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    token_rows = []
    for row in obs.astype(np.float32):
        # The exported encoder has a fixed batch dimension of 1.
        token = session.run([output_name], {input_name: row.reshape(1, -1)})[0]
        token_rows.append(np.asarray(token[0], dtype=np.float32))

    tokens = np.stack(token_rows, axis=0)
    if tokens.ndim != 2 or tokens.shape[1] != ENCODER_OUTPUT_DIM:
        raise ValueError(f"Unexpected token shape {tokens.shape}, expected [T, {ENCODER_OUTPUT_DIM}]")
    return tokens


def save_token_csv(path, tokens):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"token_{i}" for i in range(tokens.shape[1])])
        writer.writerows(tokens.tolist())


def process_motion_dir(motion_dir, encoder_session, encoder_model, output_name, save_csv, base_quat):
    joint_pos = load_csv_matrix(motion_dir / "joint_pos.csv")
    joint_vel = load_csv_matrix(motion_dir / "joint_vel.csv")
    body_quat = load_csv_matrix(motion_dir / "body_quat.csv").reshape(joint_pos.shape[0], -1, 4)
    root_quat = normalize_quat(body_quat[:, 0, :])

    if joint_pos.shape != joint_vel.shape:
        raise ValueError(f"joint_pos/joint_vel shape mismatch in {motion_dir}")
    if root_quat.shape[0] != joint_pos.shape[0]:
        raise ValueError(f"body_quat frame count mismatch in {motion_dir}")

    obs = build_g1_encoder_observations(joint_pos, joint_vel, root_quat, base_quat)
    tokens = encode_tokens(encoder_session, obs)

    output_path = motion_dir / output_name
    np.savez_compressed(
        output_path,
        token_state=tokens,
        frame_indices=np.arange(tokens.shape[0], dtype=np.int64),
        motion_length=np.asarray(tokens.shape[0], dtype=np.int64),
        encoder_model=str(Path(encoder_model).resolve()),
        encoder_input_dim=np.asarray(ENCODER_INPUT_DIM, dtype=np.int64),
        encoder_output_dim=np.asarray(ENCODER_OUTPUT_DIM, dtype=np.int64),
        encoder_mode=np.asarray(int(ENCODE_MODE_G1), dtype=np.int64),
        future_frames=np.asarray(FUTURE_FRAMES, dtype=np.int64),
        future_step=np.asarray(FUTURE_STEP, dtype=np.int64),
        window_policy="clamp_to_clip",
        feature_schema={
            "filled_observations": [
                "encoder_mode_4",
                "motion_joint_positions_10frame_step5",
                "motion_joint_velocities_10frame_step5",
                "motion_anchor_orientation_10frame_step5",
            ],
            "obs_layout": OBS_LAYOUT,
            "unused_observations": "zero_filled",
        },
    )

    if save_csv:
        save_token_csv(output_path.with_suffix(".csv"), tokens)

    return tokens.shape


def parse_base_quat(value):
    parts = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--base-quat must contain 4 comma-separated values")
    return np.asarray(parts, dtype=np.float64)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Encode reference motions into per-frame 64-D robot motion tokens.",
    )
    parser.add_argument(
        "input",
        help="Motion directory or parent directory containing motion subdirectories.",
    )
    parser.add_argument(
        "--encoder-model",
        default="gear_sonic_deploy/policy/release/model_encoder.onnx",
        help="Path to model_encoder.onnx.",
    )
    parser.add_argument(
        "--output-name",
        default="motion_token.npz",
        help="Token NPZ filename to create inside each motion directory.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save a CSV next to the token NPZ.",
    )
    parser.add_argument(
        "--base-quat",
        type=parse_base_quat,
        default=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        help="Offline base quaternion in w,x,y,z. Default: 1,0,0,0.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    encoder_model = Path(args.encoder_model)
    if not encoder_model.exists():
        parser.error(f"Encoder model not found: {encoder_model}")

    motion_dirs = discover_motion_dirs(args.input)
    if not motion_dirs:
        parser.error(f"No motion directories found under: {args.input}")

    print("Robot Motion Token Encoder")
    print("==========================")
    print(f"Input: {Path(args.input).resolve()}")
    print(f"Encoder: {encoder_model.resolve()}")
    print(f"Motions: {len(motion_dirs)}")
    print(f"Future window: {FUTURE_FRAMES} frames, step {FUTURE_STEP}, clamp_to_clip")
    encoder_session = create_encoder_session(encoder_model)

    success = 0
    for motion_dir in motion_dirs:
        print(f"\nProcessing: {motion_dir.name}")
        try:
            shape = process_motion_dir(
                motion_dir,
                encoder_session,
                encoder_model,
                args.output_name,
                args.save_csv,
                args.base_quat,
            )
        except Exception as exc:
            print(f"  Failed: {exc}")
            continue
        print(f"  Saved {args.output_name}: {shape[0]} frames x {shape[1]} dims")
        success += 1

    print(f"\nEncoded {success}/{len(motion_dirs)} motions")


if __name__ == "__main__":
    main()
