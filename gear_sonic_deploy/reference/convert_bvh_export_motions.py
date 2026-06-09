#!/usr/bin/env python3
"""
Convert SOMA retargeter bvh_export NPZ files to GR00T deployment reference CSVs.

The input NPZ files are expected to contain unitree_g1 retargeted arrays:
robot_joint_pos, robot_joint_vel, robot_body_pos, robot_body_quat,
robot_body_lin_vel, and robot_body_ang_vel.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np

from convert_motions import convert_single_motion, create_summary_file


DEFAULT_BODY_INDEXES = np.array(
    [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29],
    dtype=np.int64,
)

G1_MUJOCO_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_ISAACLAB_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

JOINT_ORDERS = {
    "isaaclab": G1_ISAACLAB_JOINT_ORDER,
    "mujoco": G1_MUJOCO_JOINT_ORDER,
}

# The NPZ robot_body_* arrays are in MuJoCo body order: pelvis + G1_MUJOCO_JOINT_ORDER links.
# Deployment metadata uses IsaacLab body indexes: pelvis + G1_ISAACLAB_JOINT_ORDER links.
ISAACLAB_BODY_INDEX_TO_NPZ_BODY_INDEX = np.array(
    [0] + [G1_MUJOCO_JOINT_ORDER.index(name) + 1 for name in G1_ISAACLAB_JOINT_ORDER],
    dtype=np.int64,
)


def parse_body_indexes(value):
    """Parse a comma-separated body index list."""
    if value is None:
        return None
    indexes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indexes:
        raise argparse.ArgumentTypeError("body index list cannot be empty")
    return np.asarray(indexes, dtype=np.int64)


def quat_to_wxyz(quat, scalar_first):
    """Return quaternions in deployment CSV order: w, x, y, z."""
    if scalar_first:
        return quat
    return quat[..., [3, 0, 1, 2]]


def decode_string_array(value):
    """Convert a numpy string array to a Python list of strings."""
    return [str(item) for item in value.tolist()]


def build_reorder_indices(source_names, target_names, label):
    """Return source column indexes that produce target_names order."""
    source_to_index = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_to_index]
    if missing:
        raise ValueError(f"Missing {label} names in source data: {missing}")
    return np.asarray([source_to_index[name] for name in target_names], dtype=np.int64)


def discover_npz_files(input_path):
    """Return sorted NPZ files from a single file or directory input."""
    input_path = os.path.abspath(input_path)
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, "*.npz")))
    if input_path.endswith(".npz"):
        return [input_path]
    raise ValueError(f"Expected a .npz file or directory, got: {input_path}")


def load_bvh_export_npz(npz_file, body_indexes, joint_order):
    """Load one bvh_export NPZ into the schema used by convert_motions.py."""
    with np.load(npz_file, allow_pickle=True) as data:
        required_keys = [
            "robot_joint_names",
            "robot_joint_pos",
            "robot_joint_vel",
            "robot_body_pos",
            "robot_body_quat",
            "robot_body_lin_vel",
            "robot_body_ang_vel",
        ]
        missing = [key for key in required_keys if key not in data.files]
        if missing:
            raise KeyError(f"Missing required NPZ fields: {missing}")

        body_pos = data["robot_body_pos"]
        num_bodies = body_pos.shape[1]
        if body_indexes is None:
            selected_body_indexes = np.arange(len(ISAACLAB_BODY_INDEX_TO_NPZ_BODY_INDEX), dtype=np.int64)
        else:
            selected_body_indexes = body_indexes
            invalid = selected_body_indexes[selected_body_indexes >= len(ISAACLAB_BODY_INDEX_TO_NPZ_BODY_INDEX)]
            if invalid.size:
                raise ValueError(
                    f"IsaacLab body indexes {invalid.tolist()} exceed supported body count "
                    f"{len(ISAACLAB_BODY_INDEX_TO_NPZ_BODY_INDEX)}"
                )
        selected_npz_body_indexes = ISAACLAB_BODY_INDEX_TO_NPZ_BODY_INDEX[selected_body_indexes]
        invalid_npz_indexes = selected_npz_body_indexes[selected_npz_body_indexes >= num_bodies]
        if invalid_npz_indexes.size:
            raise ValueError(
                f"Mapped NPZ body indexes {invalid_npz_indexes.tolist()} exceed available body count {num_bodies}"
            )

        source_joint_names = decode_string_array(data["robot_joint_names"])
        target_joint_names = JOINT_ORDERS[joint_order]
        joint_reorder = build_reorder_indices(source_joint_names, target_joint_names, "joint")

        scalar_first = bool(data["scalar_first"].item()) if "scalar_first" in data.files else False
        body_quat_wxyz = quat_to_wxyz(data["robot_body_quat"], scalar_first)
        num_frames = int(data["num_frames"].item()) if "num_frames" in data.files else body_pos.shape[0]

        return {
            "joint_pos": data["robot_joint_pos"][:, joint_reorder].astype(np.float32),
            "joint_vel": data["robot_joint_vel"][:, joint_reorder].astype(np.float32),
            "body_pos_w": body_pos[:, selected_npz_body_indexes, :].astype(np.float32),
            "body_quat_w": body_quat_wxyz[:, selected_npz_body_indexes, :].astype(np.float32),
            "body_lin_vel_w": data["robot_body_lin_vel"][:, selected_npz_body_indexes, :].astype(np.float32),
            "body_ang_vel_w": data["robot_body_ang_vel"][:, selected_npz_body_indexes, :].astype(np.float32),
            "_body_indexes": selected_body_indexes.astype(np.int64),
            "time_step_total": np.asarray(num_frames, dtype=np.int64),
            "fps": np.asarray(data["fps"].item() if "fps" in data.files else 50, dtype=np.int64),
            "joint_order": joint_order,
            "target_joint_names": target_joint_names,
            "source_joint_names": source_joint_names,
            "joint_reorder_indices": joint_reorder,
            "npz_body_indexes": selected_npz_body_indexes.astype(np.int64),
            "source_file": os.path.abspath(npz_file),
        }


def convert_bvh_export(input_path, output_dir, body_indexes, joint_order):
    """Convert one NPZ file or a directory of NPZ files."""
    npz_files = discover_npz_files(input_path)
    if not npz_files:
        print(f"No .npz files found in: {input_path}")
        return False, 0, None, None

    os.makedirs(output_dir, exist_ok=True)

    print("G1 bvh_export Motion Converter")
    print("==============================")
    print(f"Input: {os.path.abspath(input_path)}")
    print(f"Output: {output_dir}/")
    print(f"Found {len(npz_files)} NPZ motion file(s)")
    if body_indexes is None:
        print("Body selection: all bodies")
    else:
        print(f"Body selection: {body_indexes.tolist()}")
    print(f"Joint output order: {joint_order}")

    converted = {}
    for npz_file in npz_files:
        motion_name = Path(npz_file).stem
        motion_output_dir = os.path.join(output_dir, motion_name)
        print(f"\nProcessing: {motion_name}")
        print(f"Creating individual folder for this motion: {motion_output_dir}")
        os.makedirs(motion_output_dir, exist_ok=True)

        try:
            motion_data = load_bvh_export_npz(npz_file, body_indexes, joint_order)
        except Exception as exc:
            print(f"  Failed to load {npz_file}: {exc}")
            continue

        if convert_single_motion(motion_name, motion_data, motion_output_dir):
            converted[motion_name] = motion_data

    if converted:
        create_summary_file(converted, output_dir)

    joint_count = None
    body_count = None
    if converted:
        first_motion = next(iter(converted.values()))
        joint_count = first_motion["joint_pos"].shape[1]
        body_count = first_motion["body_pos_w"].shape[1]

    print(f"\nSuccessfully converted {len(converted)}/{len(npz_files)} motions")
    print(f"Output files saved to: {output_dir}/")
    return bool(converted), len(converted), joint_count, body_count


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert SOMA bvh_export NPZ motions to deployment reference CSV folders.",
    )
    parser.add_argument("input", help="Path to one .npz file or a directory containing .npz files.")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output base directory. Defaults to reference/<input_name>.",
    )
    parser.add_argument(
        "--all-bodies",
        action="store_true",
        help="Export every robot body from the NPZ instead of the default 14 deployment bodies.",
    )
    parser.add_argument(
        "--body-indexes",
        type=parse_body_indexes,
        help="Comma-separated IsaacLab body indexes to export, e.g. 0,4,10,18.",
    )
    parser.add_argument(
        "--joint-order",
        choices=sorted(JOINT_ORDERS),
        default="isaaclab",
        help="Output order for joint_pos.csv and joint_vel.csv. Default: isaaclab.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.all_bodies and args.body_indexes is not None:
        parser.error("--all-bodies and --body-indexes are mutually exclusive")

    input_path = os.path.abspath(args.input)
    input_name = Path(input_path).stem if input_path.endswith(".npz") else Path(input_path).name
    output_dir = args.output
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent / input_name)

    if args.all_bodies:
        body_indexes = None
    elif args.body_indexes is not None:
        body_indexes = args.body_indexes
    else:
        body_indexes = DEFAULT_BODY_INDEXES

    success, motion_count, joint_count, body_count = convert_bvh_export(
        input_path, output_dir, body_indexes, args.joint_order
    )
    if success:
        print("\nConversion completed successfully")
        print(f"- Motions: {motion_count}")
        print(f"- Joints: {joint_count}")
        print(f"- Body parts: {body_count}")
    else:
        print("\nConversion failed")


if __name__ == "__main__":
    main()
