# Offline SMPL Reference For Deploy

This document records the offline SMPL reference path added for
`g1_deploy_onnx_ref`. It is intended as a future engineering reference for
feeding saved SMPL data into the SMPL encoder mode and running inference in the
MuJoCo/deploy loop.

## Purpose

The deploy policy can use an encoder with multiple input modes. For offline
SMPL motion replay, the relevant mode is `smpl` in
`policy/release/observation_config.yaml`. That mode requires:

- `encoder_mode_4`
- `smpl_joints_10frame_step1`
- `smpl_anchor_orientation_10frame_step1`
- `motion_joint_positions_wrists_10frame_step1`

The added offline path converts an SMPL `.npz` file into the existing deploy
reference-motion CSV layout, then lets the normal reference reader and encoder
pipeline consume it.

## Input Contract

The converter expects an SMPL `.npz` with these keys:

- `global_orient`: shape `(T, 3)`, axis-angle root orientation
- `body_pose`: shape `(T, 69)` or larger, axis-angle body pose
- `transl`: shape `(T, 3)`, loaded for validation and future use
- `betas`: shape `(T, 10)` or `(10,)`, optional

Only the first 63 columns of `body_pose` are used for the deploy SMPL encoder
path, matching the existing Pico processing path.

## Conversion Pipeline

The converter is:

```text
reference/convert_smpl_npz_to_reference.py
```

The important implementation classes are:

- `SmplNpzDataSource`: loads and validates the `.npz`, then samples frames onto
  the deploy target frame rate.
- `PicoSmplReferenceProcessor`: converts the raw SMPL parameters using the same
  root-orientation convention as the Pico online path.
- `DeployReferenceCsvWriter`: writes a deploy-compatible motion folder.
- `SmplReferenceConversionApp`: wires the source, processor, and writer
  together for the CLI.

The SMPL processing intentionally mirrors the Pico path:

1. Convert `global_orient` axis-angle to quaternion.
2. Convert the SMPL root from y-up to deploy z-up with `smpl_root_ytoz_up`.
3. Convert the adjusted root back to axis-angle.
4. Run `compute_human_joints` using `body_pose[:, :63]` and the adjusted root.
5. Remove the SMPL base rotation with `remove_smpl_base_rot`.
6. Rotate joints into root-local coordinates with the inverse processed root
   quaternion.

The resulting `smpl_joint.csv` therefore contains root-local SMPL joint
positions, not raw world-space joints from an external viewer.

## Frame Rate Handling

The Nymeria SMPL data used during development is 240 Hz, while deploy reference
playback expects 50 Hz. The converter defaults to:

```text
source_fps = 240
target_fps = 50
```

It does nearest-frame time sampling:

```text
source_index = round(target_frame_index * source_fps / target_fps)
```

For 240 Hz to 50 Hz, the first sampled source frames are approximately:

```text
0, 5, 10, 14, 19, 24, 29, ...
```

This is not `stride=5`, which would produce 48 Hz playback. It is also not
interpolation. If a future project needs higher-fidelity resampling, root and
body rotations should be interpolated with quaternion SLERP and translations
with linear interpolation.

`--max-frames` means maximum output frames at `--target-fps`. For example,
`--max-frames 1000 --target-fps 50` produces about 20 seconds of reference.

## Output Motion Folder

A generated motion folder contains:

- `smpl_joint.csv`: `(frames, 72)`, 24 SMPL joints times xyz
- `smpl_pose.csv`: `(frames, 63)`, 21 SMPL body joint axis-angle vectors
- `body_quat.csv`: `(frames, 4)`, processed root quaternion in wxyz order
- `joint_pos.csv`: `(frames, 29)`, robot joint placeholders
- `joint_vel.csv`: `(frames, 29)`, robot joint velocity placeholders
- `body_pos.csv`: `(frames, 3)`, currently zeros
- `body_lin_vel.csv`: `(frames, 3)`, currently zeros
- `body_ang_vel.csv`: `(frames, 3)`, currently zeros
- `metadata.txt`
- `info.txt`

For the current offline SMPL use case, the six G1 wrist joint values required
by `motion_joint_positions_wrists_10frame_step1` are zero placeholders inside
`joint_pos.csv`. The indices are:

```text
23, 24, 25, 26, 27, 28
```

## Current Example

The tested Nymeria input is:

```bash
/home/hpx/HPX_LOCO_2/mimic_baseline/nymeria_parse/out/batch/20230607_s0_james_johnson_act0_e72nhq/smpl/nymeria_smpl.npz
```

Generate a 1000-frame, 50 Hz reference:

```bash
cd /home/hpx/HPX_LOCO_2/GR00T-WholeBodyControl

python3 gear_sonic_deploy/reference/convert_smpl_npz_to_reference.py \
  --npz /home/hpx/HPX_LOCO_2/mimic_baseline/nymeria_parse/out/batch/20230607_s0_james_johnson_act0_e72nhq/smpl/nymeria_smpl.npz \
  --output-root gear_sonic_deploy/reference/offline_smpl \
  --motion-name nymeria_smpl_1000 \
  --max-frames 1000 \
  --source-fps 240 \
  --target-fps 50
```

The generated folder is:

```text
gear_sonic_deploy/reference/offline_smpl/nymeria_smpl_1000
```

`info.txt` records the sampling metadata, for example:

```text
frames: 1000
source_fps: 240.0
target_fps: 50.0
source_frame_first: 0
source_frame_last: 4795
encoder_mode: smpl (2)
```

## Deploy Integration

`src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` now detects offline
SMPL reference motions during initialization. If a loaded motion contains at
least 24 SMPL joints and the encoder is available, that motion is assigned:

```text
encode_mode = 2
```

The expected startup log is:

```text
Motion 'nymeria_smpl_1000' encode_mode set to: 2 (offline SMPL reference detected)
```

This keeps the planner/default mode unchanged while allowing the offline SMPL
reference motion to route through the SMPL encoder requirements in
`observation_config.yaml`.

## Running With MuJoCo

Terminal 1, start the simulator from the repository root:

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

Terminal 2, start deploy:

```bash
cd /home/hpx/HPX_LOCO_2/GR00T-WholeBodyControl/gear_sonic_deploy

./target/release/g1_deploy_onnx_ref \
  lo \
  policy/release/model_decoder.onnx \
  reference/offline_smpl \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --input-type keyboard \
  --output-type zmq \
  --disable-crc-check
```

Useful keyboard flow:

- `]`: start control
- `T`: play the current reference motion
- `R`: restart current motion at frame 0
- `N` / `P`: switch motion if multiple reference folders exist
- `O`: stop motion playback

## Verification Commands

Python tests:

```bash
cd /home/hpx/HPX_LOCO_2/GR00T-WholeBodyControl
python3 -m pytest \
  gear_sonic_deploy/reference/tests/test_convert_smpl_npz_to_reference.py \
  gear_sonic_deploy/reference/tests/test_visualize_smpl_reference.py \
  -q
```

Build deploy:

```bash
cd /home/hpx/HPX_LOCO_2/GR00T-WholeBodyControl/gear_sonic_deploy
just build
```

Smoke test without MuJoCo:

```bash
cd /home/hpx/HPX_LOCO_2/GR00T-WholeBodyControl/gear_sonic_deploy

timeout 25s ./target/release/g1_deploy_onnx_ref \
  lo \
  policy/release/model_decoder.onnx \
  reference/offline_smpl \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --input-type keyboard \
  --output-type zmq \
  --disable-crc-check
```

Without MuJoCo running, the process is expected to eventually print
`LowState is not available, waiting for robot to be ready`. The useful smoke
test signal is that the reference loads, encoder dimensions match, and the
motion switches to `encode_mode = 2`.

## Future Work Notes

- Add interpolation if nearest-frame sampling becomes too coarse for fast
  motions. Use SLERP for rotations instead of linear interpolation on
  axis-angle vectors.
- Replace zero wrist placeholders with real retargeted G1 wrist joint values if
  hand or wrist behavior becomes important.
- Keep visualization separate from encoder input semantics. A viewer may need
  world-space joints and ground alignment, while the encoder currently expects
  Pico-style root-local joints.
- If a new dataset has a different source frame rate, pass the explicit
  `--source-fps` value instead of relying on the Nymeria default.
