import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convert_smpl_npz_to_reference import (  # noqa: E402
    ConversionConfig,
    DeployReferenceCsvWriter,
    SmplNpzDataSource,
    SmplReferenceData,
)


def _read_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def test_deploy_reference_writer_creates_required_smpl_mode_files(tmp_path):
    frames = 3
    data = SmplReferenceData(
        name="unit_motion",
        smpl_joints_local=np.arange(frames * 24 * 3, dtype=np.float32).reshape(frames, 24, 3),
        smpl_pose=np.ones((frames, 21, 3), dtype=np.float32),
        body_quat_w=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (frames, 1)),
        joint_pos=np.zeros((frames, 29), dtype=np.float32),
        joint_vel=np.zeros((frames, 29), dtype=np.float32),
    )

    motion_dir = DeployReferenceCsvWriter(tmp_path).write(data)

    assert motion_dir == tmp_path / "unit_motion"
    assert (motion_dir / "smpl_joint.csv").exists()
    assert (motion_dir / "smpl_pose.csv").exists()
    assert (motion_dir / "body_quat.csv").exists()
    assert (motion_dir / "joint_pos.csv").exists()
    assert (motion_dir / "joint_vel.csv").exists()
    assert (motion_dir / "metadata.txt").exists()
    assert (motion_dir / "info.txt").exists()

    smpl_joint_rows = _read_csv(motion_dir / "smpl_joint.csv")
    smpl_pose_rows = _read_csv(motion_dir / "smpl_pose.csv")
    joint_pos_rows = _read_csv(motion_dir / "joint_pos.csv")

    assert smpl_joint_rows.shape == (frames, 72)
    assert smpl_pose_rows.shape == (frames, 63)
    assert joint_pos_rows.shape == (frames, 29)
    np.testing.assert_allclose(smpl_joint_rows[0, :3], [0.0, 1.0, 2.0])


def test_deploy_reference_writer_preserves_zero_wrist_joint_placeholders(tmp_path):
    frames = 2
    data = SmplReferenceData(
        name="wrist_zero",
        smpl_joints_local=np.zeros((frames, 24, 3), dtype=np.float32),
        smpl_pose=np.zeros((frames, 21, 3), dtype=np.float32),
        body_quat_w=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (frames, 1)),
        joint_pos=np.zeros((frames, 29), dtype=np.float32),
        joint_vel=np.zeros((frames, 29), dtype=np.float32),
    )

    motion_dir = DeployReferenceCsvWriter(tmp_path).write(data)
    joint_pos_rows = _read_csv(motion_dir / "joint_pos.csv")

    wrist_indices = [23, 24, 25, 26, 27, 28]
    np.testing.assert_allclose(joint_pos_rows[:, wrist_indices], 0.0)


def test_smpl_npz_data_source_resamples_240hz_to_50hz_by_time(tmp_path):
    frames = 30
    source_indices = np.arange(frames, dtype=np.float32)
    npz_path = tmp_path / "source_240hz.npz"
    np.savez(
        npz_path,
        global_orient=np.repeat(source_indices[:, None], 3, axis=1),
        body_pose=np.repeat(source_indices[:, None], 69, axis=1),
        transl=np.repeat(source_indices[:, None], 3, axis=1),
        betas=np.repeat(source_indices[:, None], 10, axis=1),
    )
    config = ConversionConfig(
        max_frames=5,
        source_fps=240.0,
        target_fps=50.0,
    )

    data = SmplNpzDataSource(npz_path, config).load()

    np.testing.assert_allclose(data.global_orient[:, 0], [0.0, 5.0, 10.0, 14.0, 19.0])
    np.testing.assert_allclose(data.body_pose[:, 0], [0.0, 5.0, 10.0, 14.0, 19.0])
