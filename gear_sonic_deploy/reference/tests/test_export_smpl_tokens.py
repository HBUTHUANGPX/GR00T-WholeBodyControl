import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convert_smpl_npz_to_reference import SmplReferenceData  # noqa: E402
from export_smpl_tokens import (  # noqa: E402
    ENCODE_MODE_SMPL,
    SMPL_FUTURE_FRAMES,
    WRIST_JOINT_INDICES,
    build_smpl_encoder_observations,
    build_arg_parser,
    default_token_output_path,
    discover_smpl_npz_files,
)
from encode_motion_tokens import OBS_LAYOUT  # noqa: E402


def _reference(frames: int = 3) -> SmplReferenceData:
    smpl_joints = np.arange(frames * 24 * 3, dtype=np.float32).reshape(frames, 24, 3)
    joint_pos = np.zeros((frames, 29), dtype=np.float32)
    for frame in range(frames):
        joint_pos[frame, WRIST_JOINT_INDICES] = np.arange(6, dtype=np.float32) + frame * 10
    return SmplReferenceData(
        name="unit_smpl",
        smpl_joints_local=smpl_joints,
        smpl_pose=np.zeros((frames, 21, 3), dtype=np.float32),
        body_quat_w=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (frames, 1)),
        joint_pos=joint_pos,
        joint_vel=np.zeros_like(joint_pos),
        source_frame_indices=np.arange(frames, dtype=np.int64),
        source_fps=50.0,
        target_fps=50.0,
    )


def test_build_smpl_encoder_observations_preserves_frame_count_and_sets_mode():
    reference = _reference(frames=3)

    obs = build_smpl_encoder_observations(reference)

    assert obs.shape == (3, 1762)
    mode_offset, mode_dim = OBS_LAYOUT["encoder_mode_4"]
    np.testing.assert_allclose(obs[:, mode_offset : mode_offset + mode_dim], [[ENCODE_MODE_SMPL, 0, 0, 0]] * 3)


def test_build_smpl_encoder_observations_uses_step1_clamped_future_windows():
    reference = _reference(frames=3)

    obs = build_smpl_encoder_observations(reference)

    smpl_offset, _ = OBS_LAYOUT["smpl_joints_10frame_step1"]
    frame_one_window = obs[1, smpl_offset : smpl_offset + SMPL_FUTURE_FRAMES * 24 * 3]
    expected_indices = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    expected = reference.smpl_joints_local[expected_indices].reshape(-1)
    np.testing.assert_allclose(frame_one_window, expected)


def test_build_smpl_encoder_observations_fills_anchor_orientation_and_wrists():
    reference = _reference(frames=3)

    obs = build_smpl_encoder_observations(reference)

    ori_offset, _ = OBS_LAYOUT["smpl_anchor_orientation_10frame_step1"]
    expected_identity_rot6d = np.tile(np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32), SMPL_FUTURE_FRAMES)
    np.testing.assert_allclose(obs[0, ori_offset : ori_offset + SMPL_FUTURE_FRAMES * 6], expected_identity_rot6d)

    wrist_offset, _ = OBS_LAYOUT["motion_joint_positions_wrists_10frame_step1"]
    expected_wrists = reference.joint_pos[[0, 1, 2, 2, 2, 2, 2, 2, 2, 2]][:, WRIST_JOINT_INDICES].reshape(-1)
    np.testing.assert_allclose(obs[0, wrist_offset : wrist_offset + SMPL_FUTURE_FRAMES * 6], expected_wrists)


def test_build_smpl_encoder_observations_can_build_global_frame_slice():
    reference = _reference(frames=4)

    obs = build_smpl_encoder_observations(reference, frame_start=2, frame_end=4)

    assert obs.shape == (2, 1762)
    smpl_offset, _ = OBS_LAYOUT["smpl_joints_10frame_step1"]
    expected_indices = [2, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    expected = reference.smpl_joints_local[expected_indices].reshape(-1)
    np.testing.assert_allclose(obs[0, smpl_offset : smpl_offset + SMPL_FUTURE_FRAMES * 24 * 3], expected)


def test_default_token_output_path_saves_next_to_source_npz():
    source = Path("/dataset/subject/session/smpl_data.npz")

    output = default_token_output_path(source)

    assert output == Path("/dataset/subject/session/smpl_data_motion_token.npz")


def test_discover_smpl_npz_files_recurses_and_ignores_non_smpl_npz():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        valid = tmp_path / "subject" / "session" / "smpl_data.npz"
        valid.parent.mkdir(parents=True)
        np.savez(
            valid,
            global_orient=np.zeros((2, 3), dtype=np.float32),
            body_pose=np.zeros((2, 69), dtype=np.float32),
            transl=np.zeros((2, 3), dtype=np.float32),
        )
        np.savez(valid.parent / "smpl_data_motion_token.npz", token_state=np.zeros((2, 64), dtype=np.float32))
        np.savez(tmp_path / "SMPL_NEUTRAL.npz", J=np.zeros((24, 3), dtype=np.float32))

        discovered = discover_smpl_npz_files(tmp_path)

        assert discovered == [valid]


def test_arg_parser_accepts_streamlined_batch_usage():
    args = build_arg_parser().parse_args(["/dataset/root", "--max-frames", "5", "--overwrite"])

    assert args.input == Path("/dataset/root")
    assert args.max_frames == 5
    assert args.overwrite is True
