from pathlib import Path

import numpy as np

from gear_sonic.utils.motrixsim_sim.human_reference import (
    HumanReferenceConfig,
    HumanReferencePlayer,
)


def _write_smpl_joint_csv(path: Path, frames: np.ndarray) -> None:
    rows = frames.reshape(frames.shape[0], -1)
    headers = [f"smpl_joint_{idx}" for idx in range(rows.shape[1])]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        np.savetxt(handle, rows, delimiter=",", fmt="%.6f")


def test_human_reference_player_loads_first_motion_folder(tmp_path):
    motion_dir = tmp_path / "motion_a"
    motion_dir.mkdir()
    frames = np.zeros((2, 24, 3), dtype=np.float32)
    frames[1, 1] = [1.0, 2.0, 3.0]
    _write_smpl_joint_csv(motion_dir / "smpl_joint.csv", frames)

    player = HumanReferencePlayer.from_config(HumanReferenceConfig(reference_path=tmp_path))

    assert player is not None
    np.testing.assert_allclose(player.frame_at_time(1.0 / 50.0)[1, :2], [1.0, 3.25])
    assert np.isclose(player.frame_at_time(1.0 / 50.0)[:, 2].min(), 0.02)


def test_human_reference_player_can_sample_explicit_deploy_frame(tmp_path):
    motion_dir = tmp_path / "motion_a"
    motion_dir.mkdir()
    frames = np.zeros((3, 24, 3), dtype=np.float32)
    frames[2, 1] = [4.0, 5.0, 6.0]
    _write_smpl_joint_csv(motion_dir / "smpl_joint.csv", frames)

    player = HumanReferencePlayer.from_config(HumanReferenceConfig(reference_path=tmp_path))

    assert player is not None
    np.testing.assert_allclose(player.frame_at_index(2)[1, :2], [4.0, 6.25])
    np.testing.assert_allclose(player.frame_at_index(5)[1, :2], [4.0, 6.25])


def test_human_reference_player_returns_none_when_disabled(tmp_path):
    player = HumanReferencePlayer.from_config(
        HumanReferenceConfig(reference_path=tmp_path, enabled=False)
    )

    assert player is None
