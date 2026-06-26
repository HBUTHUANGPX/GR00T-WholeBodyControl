import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_nymeria_rgb_smpl_tokens import (  # noqa: E402
    FUTURE_STEP_NS,
    build_rgb_aligned_window_plan,
    default_output_path,
    discover_sequence_inputs,
    interpolate_smpl_motion,
    iter_frame_chunks,
)


class NymeriaRgbSmplTokenExportTest(unittest.TestCase):
    def test_window_plan_starts_at_rgb_frame_before_first_smpl_and_keeps_rgb_tail(self):
        smpl_times = np.asarray([100, 200, 300], dtype=np.int64)
        rgb_times = np.asarray([0, 90, 110, 320, 350], dtype=np.int64)
        rgb_indices = np.asarray([10, 11, 12, 13, 14], dtype=np.int32)

        plan = build_rgb_aligned_window_plan(
            smpl_times,
            rgb_times,
            rgb_indices,
            future_frames=3,
            future_step_ns=20,
        )

        np.testing.assert_array_equal(plan.rgb_positions, [1, 2, 3, 4])
        np.testing.assert_array_equal(plan.rgb_frame_indices, [11, 12, 13, 14])
        np.testing.assert_array_equal(plan.rgb_relative_timestamps_ns, [90, 110, 320, 350])
        np.testing.assert_array_equal(
            plan.sample_relative_timestamps_ns,
            [
                [100, 110, 130],
                [110, 130, 150],
                [300, 300, 300],
                [300, 300, 300],
            ],
        )
        self.assertTrue(plan.clamped_sample_mask[0, 0])
        self.assertTrue(plan.clamped_sample_mask[2].all())
        self.assertTrue(plan.clamped_sample_mask[3].all())

    def test_window_plan_uses_50hz_future_step_by_default(self):
        smpl_times = np.asarray([0, 100_000_000], dtype=np.int64)
        rgb_times = np.asarray([0], dtype=np.int64)

        plan = build_rgb_aligned_window_plan(smpl_times, rgb_times)

        np.testing.assert_array_equal(
            plan.sample_relative_timestamps_ns[0, :4],
            [0, FUTURE_STEP_NS, FUTURE_STEP_NS * 2, FUTURE_STEP_NS * 3],
        )

    def test_interpolate_smpl_motion_uses_linear_translation_and_slerp_rotvec(self):
        smpl_times = np.asarray([0, 1_000_000_000], dtype=np.int64)
        sample_times = np.asarray([[500_000_000]], dtype=np.int64)
        global_orient = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, math.pi / 2.0],
            ],
            dtype=np.float32,
        )
        body_pose = np.zeros((2, 69), dtype=np.float32)
        body_pose[1, 2] = math.pi / 2.0
        transl = np.asarray([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]], dtype=np.float32)
        betas = np.asarray([[0.0] * 10, [1.0] * 10], dtype=np.float32)

        sampled = interpolate_smpl_motion(
            smpl_times,
            global_orient,
            body_pose,
            transl,
            betas,
            sample_times,
        )

        np.testing.assert_allclose(sampled.global_orient[0, 0], [0.0, 0.0, math.pi / 4.0], atol=1e-5)
        np.testing.assert_allclose(sampled.body_pose[0, 0, :3], [0.0, 0.0, math.pi / 4.0], atol=1e-5)
        np.testing.assert_allclose(sampled.transl[0, 0], [5.0, 10.0, 15.0], atol=1e-6)
        np.testing.assert_allclose(sampled.betas[0, 0], [0.5] * 10, atol=1e-6)
        np.testing.assert_array_equal(sampled.left_indices, [[0]])
        np.testing.assert_array_equal(sampled.right_indices, [[1]])
        np.testing.assert_allclose(sampled.alpha, [[0.5]], atol=1e-6)

    def test_discover_sequence_inputs_finds_sequences_under_root_and_uses_token_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ny_batch"
            sequence_a = root / "seq_a"
            sequence_b = root / "nested" / "seq_b"
            for sequence in (sequence_a, sequence_b):
                (sequence / "smpl").mkdir(parents=True)
                (sequence / "head_video").mkdir()
                (sequence / "smpl" / "nymeria_smpl.npz").write_bytes(b"placeholder")
                (sequence / "head_video" / "timestamps.npz").write_bytes(b"placeholder")
            (root / "not_a_sequence" / "smpl").mkdir(parents=True)
            (root / "not_a_sequence" / "smpl" / "nymeria_smpl.npz").write_bytes(b"placeholder")

            jobs = discover_sequence_inputs(root)

            self.assertEqual([job.sequence_dir for job in jobs], [sequence_b, sequence_a])
            self.assertEqual(
                [job.output_path for job in jobs],
                [sequence_b / "token" / "token.npz", sequence_a / "token" / "token.npz"],
            )

    def test_default_output_path_is_sequence_token_folder(self):
        sequence = Path("/data/ny_batch/seq_a")

        self.assertEqual(default_output_path(sequence), sequence / "token" / "token.npz")

    def test_iter_frame_chunks_updates_progress_by_frame_count(self):
        class RecordingProgress:
            def __init__(self):
                self.updates = []

            def update(self, count):
                self.updates.append(count)

        progress = RecordingProgress()

        ranges = list(iter_frame_chunks(total=5, chunk_size=2, progress=progress))

        self.assertEqual(ranges, [(0, 2), (2, 4), (4, 5)])
        self.assertEqual(progress.updates, [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
