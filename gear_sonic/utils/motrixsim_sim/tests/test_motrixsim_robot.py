import numpy as np

from gear_sonic.utils.motrixsim_sim.robot import MotrixG1Robot
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig


def test_motrixsim_g1_robot_preserves_body_and_hand_joint_groups():
    import motrixsim as mx

    config = SimLoopConfig(simulator="motrixsim", enable_onscreen=False).load_wbc_yaml()
    model = mx.load_model(config["ROBOT_SCENE"])
    data = mx.SceneData(model)

    robot = MotrixG1Robot(model, config)

    assert len(robot.body_joint_names) == 29
    assert len(robot.left_hand_joint_names) == 7
    assert len(robot.right_hand_joint_names) == 7
    assert robot.body_q(data).shape == (29,)
    assert robot.body_dq(data).shape == (29,)
    np.testing.assert_allclose(robot.floating_base_pose_wxyz(data)[3:7], [1.0, 0.0, 0.0, 0.0])


def test_motrixsim_g1_robot_publishes_local_base_angular_velocity():
    import motrixsim as mx

    config = SimLoopConfig(simulator="motrixsim", enable_onscreen=False).load_wbc_yaml()
    model = mx.load_model(config["ROBOT_SCENE"])
    data = mx.SceneData(model)
    robot = MotrixG1Robot(model, config)
    floating_base = model.floating_bases[0]
    floating_base.set_global_linear_velocity(data, np.array([1.0, 2.0, 3.0]))
    floating_base.set_local_angular_velocity(data, np.array([0.4, 0.5, 0.6]))

    velocity = robot.floating_base_velocity(data)

    np.testing.assert_allclose(velocity[:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(velocity[3:], [0.4, 0.5, 0.6])
