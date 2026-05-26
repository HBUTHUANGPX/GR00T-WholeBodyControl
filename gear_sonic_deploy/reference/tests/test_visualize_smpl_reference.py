import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualize_smpl_reference import (
    GroundAlignmentTransform,
    IdentityCoordinateTransform,
    MujocoSkeletonSceneBuilder,
    ReferenceSkeletonSource,
    SkeletonFrame,
    SkeletonFrameValidator,
    SmplToMujocoCoordinateTransform,
)


def _write_csv(path: Path, array: np.ndarray) -> None:
    headers = [f"col_{idx}" for idx in range(array.reshape(array.shape[0], -1).shape[1])]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        np.savetxt(handle, array.reshape(array.shape[0], -1), delimiter=",", fmt="%.6f")


def test_reference_source_loads_smpl_joints_and_body_quaternions(tmp_path):
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    smpl_joints = np.zeros((2, 24, 3), dtype=np.float32)
    smpl_joints[1, 1] = [1.0, 2.0, 3.0]
    body_quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1))
    _write_csv(motion_dir / "smpl_joint.csv", smpl_joints)
    _write_csv(motion_dir / "body_quat.csv", body_quat)

    source = ReferenceSkeletonSource(motion_dir)
    frames = source.load()

    assert len(frames) == 2
    assert frames[0].joint_positions.shape == (24, 3)
    np.testing.assert_allclose(frames[1].joint_positions[1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(frames[0].root_quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])


def test_validator_detects_root_local_reference():
    frames = [
        SkeletonFrame(
            joint_positions=np.zeros((24, 3), dtype=np.float32),
            root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            frame_index=0,
            label="unit",
        )
    ]

    result = SkeletonFrameValidator().summarize(frames)

    assert result["root_max_abs"] == 0.0
    assert result["is_root_local"] is True


def test_mujoco_scene_builder_creates_valid_z_up_scene():
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[:, 2] = np.linspace(0.0, 1.0, 24)
    joints[:, 0] = np.linspace(0.0, 0.2, 24)
    frame = SkeletonFrame(
        joint_positions=joints,
        root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        frame_index=0,
        label="unit",
    )

    xml = MujocoSkeletonSceneBuilder().build(frame)

    assert 'name="axis_x_front"' in xml
    assert 'name="axis_y_left"' in xml
    assert 'name="axis_z_up"' in xml
    assert 'name="smpl_joint_00"' in xml

    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nbody >= 25


def test_smpl_to_mujoco_transform_maps_human_axes_to_robot_axes():
    transform = SmplToMujocoCoordinateTransform()
    points = np.array(
        [
            [0.0, 1.0, 0.0],  # SMPL up
            [0.0, 0.0, -1.0],  # SMPL front
            [-1.0, 0.0, 0.0],  # SMPL left
        ],
        dtype=np.float32,
    )

    transformed = transform.transform_points(points)

    np.testing.assert_allclose(transformed[0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(transformed[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(transformed[2], [0.0, 1.0, 0.0])


def test_smpl_to_mujoco_transform_can_transform_a_frame():
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[0] = [1.0, 2.0, 3.0]
    frame = SkeletonFrame(
        joint_positions=joints,
        root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        frame_index=7,
        label="raw",
    )

    transformed = SmplToMujocoCoordinateTransform().transform_frame(frame)

    np.testing.assert_allclose(transformed.joint_positions[0], [-3.0, -1.0, 2.0])
    assert transformed.frame_index == 7
    assert transformed.label == "raw_mujoco"


def test_ground_alignment_transform_raises_lowest_joint_to_clearance():
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[0] = [0.0, 0.0, 0.4]
    joints[1] = [0.0, 0.0, -0.2]
    frame = SkeletonFrame(
        joint_positions=joints,
        root_quaternion_wxyz=None,
        frame_index=3,
        label="standing",
    )

    transformed = GroundAlignmentTransform(IdentityCoordinateTransform(), clearance=0.03).transform_frame(frame)

    np.testing.assert_allclose(transformed.joint_positions[:, 2].min(), 0.03)
    np.testing.assert_allclose(transformed.joint_positions[0], [0.0, 0.0, 0.63])
    assert transformed.frame_index == 3
    assert transformed.label == "standing_ground"
