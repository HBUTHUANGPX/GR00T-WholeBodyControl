# MotrixSim Backend

This package implements a MotrixSim version of the Gear Sonic simulation loop.
It is selected through the existing `run_sim_loop.py` entry point:

```bash
source .venv_sim/bin/activate

python gear_sonic/scripts/run_sim_loop.py \
  --simulator motrixsim \
  --interface lo
```

For headless smoke tests:

```bash
python gear_sonic/scripts/run_sim_loop.py \
  --simulator motrixsim \
  --interface lo \
  --no-enable-onscreen \
  --no-enable-offscreen \
  --no-enable-image-publish
```

## Architecture

The implementation mirrors `gear_sonic/utils/mujoco_sim`:

- `base_sim.py`: MotrixSim `DefaultEnv` and `BaseSimulator`.
- `robot.py`: G1 joint/link/actuator accessor preserving the current Gear Sonic
  G1 MJCF names and body/hand joint ordering.
- `human_reference.py`: optional deploy-format SMPL reference skeleton
  visualization for the MotrixSim GUI.

The deploy binary is not changed. Communication still flows through the
Unitree SDK bridge:

```text
g1_deploy_onnx_ref
  -> rt/lowcmd
  -> MotrixSim actuator controls
  -> MotrixSim physics
  -> rt/lowstate / rt/secondary_imu / rt/odostate
  -> g1_deploy_onnx_ref observations
```

## Robot Model

MotrixSim loads the same configured Gear Sonic G1 scene:

```text
gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml
```

The backend does not use MotrixSim's example G1 asset, because deploy depends
on the current project's joint names and order.

## Human Skeleton Visualization

When available, the GUI draws a deploy-format SMPL skeleton from:

```text
gear_sonic_deploy/reference/offline_smpl
```

This can be overridden with:

```bash
--motrixsim-human-reference-path /path/to/reference/root_or_motion_dir
```

Disable it with:

```bash
--no-motrixsim-show-human-reference
```

The skeleton visualization is intentionally independent from the encoder input
path. It reads `smpl_joint.csv` for visual inspection while the deploy process
continues to read references and run encoder/decoder inference as before.

## Image Publishing

MotrixSim GUI rendering is implemented. Offscreen framebuffer readback/image
publishing is currently left as a compatibility stub because the public examples
do not expose a stable image readback API. If `--enable-image-publish` is set,
the backend prints a warning and continues with GUI/physics simulation.

## Verification

Run MotrixSim unit tests:

```bash
source .venv_sim/bin/activate
python -m pytest gear_sonic/utils/motrixsim_sim/tests -q
```

Robot reference smoke:

```bash
# Terminal 1
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
  --simulator motrixsim \
  --interface lo \
  --no-enable-onscreen \
  --no-enable-offscreen \
  --no-enable-image-publish \
  --no-motrixsim-show-human-reference

# Terminal 2
cd gear_sonic_deploy
timeout 26s ./target/release/g1_deploy_onnx_ref \
  lo \
  policy/release/model_decoder.onnx \
  reference/example \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --input-type keyboard \
  --output-type zmq \
  --disable-crc-check
```

SMPL/human reference smoke:

```bash
# Terminal 1
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
  --simulator motrixsim \
  --interface lo \
  --no-enable-onscreen \
  --no-enable-offscreen \
  --no-enable-image-publish

# Terminal 2
cd gear_sonic_deploy
timeout 30s ./target/release/g1_deploy_onnx_ref \
  lo \
  policy/release/model_decoder.onnx \
  reference/offline_smpl \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --input-type keyboard \
  --output-type zmq \
  --disable-crc-check
```

Expected signal:

- Deploy reaches `Init Done`.
- No repeated `LowState is not available` messages.
- `reference/example` motions keep `encode_mode = 0`.
- `reference/offline_smpl` loads `nymeria_smpl_1000` and switches to
  `encode_mode = 2`.
