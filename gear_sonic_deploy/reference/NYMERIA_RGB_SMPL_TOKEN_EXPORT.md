# Nymeria RGB-Aligned SMPL Token Export

This document explains `reference/export_nymeria_rgb_smpl_tokens.py`.

The script exports one Sonic FSQ token for each selected Nymeria RGB frame.
The RGB frame rate is not used as the encoder frame rate. Each RGB frame is
only an anchor timestamp. For that anchor, the script builds a strict 50 Hz
SMPL future window and sends that window to `model_encoder.onnx`.

## Input Layout

The recommended input is a root directory containing one or more converted
Nymeria sequence directories:

```text
<root>/
├── <sequence-a>/
│   ├── smpl/
│   │   └── nymeria_smpl.npz
│   └── head_video/
│       └── timestamps.npz
└── <sequence-b>/
    ├── smpl/
    │   └── nymeria_smpl.npz
    └── head_video/
        └── timestamps.npz
```

The script recursively discovers every directory that has this sequence layout:

```text
<sequence-dir>/
├── smpl/
│   └── nymeria_smpl.npz
└── head_video/
    └── timestamps.npz
```

`nymeria_smpl.npz` must contain:

- `global_orient`: SMPL root axis-angle, shape `(T, 3)`
- `body_pose`: SMPL body axis-angle, shape `(T, 69)`
- `transl`: SMPL translation, shape `(T, 3)`
- `betas`: SMPL shape, shape `(T, 10)`, `(1, 10)`, or `(10,)`
- `relative_timestamps_ns`, or `timestamps_ns` plus `time_zero_ns`
- `time_zero_ns`

`head_video/timestamps.npz` must contain:

- `rgb_relative_timestamps_ns`
- `rgb_frame_indices`
- `time_zero_ns`

Both files should describe the same Nymeria time axis:

```text
time_domain = "time_code"
time_zero_source = "recording_head/rgb/frame_0"
```

## Time Alignment

Nymeria RGB and SMPL are aligned by relative nanosecond timestamps. RGB frame
zero is the first RGB frame. SMPL may start later than RGB.

Let:

```text
S[i] = SMPL relative timestamp for source SMPL frame i
R[k] = RGB relative timestamp for RGB frame k
```

The script starts from the RGB frame immediately before the first SMPL frame:

```text
k0 = searchsorted(R, S[0], side="right") - 1
```

If no RGB frame exists before `S[0]`, it starts from RGB frame zero.

## 50 Hz SMPL Window

For each selected RGB frame `k`, the RGB timestamp is only an anchor:

```text
anchor = R[k]
```

The encoder input still uses a 10-frame, 50 Hz SMPL future window:

```text
anchor + 0 ms
anchor + 20 ms
anchor + 40 ms
...
anchor + 180 ms
```

In nanoseconds:

```text
sample_time[n] = R[k] + n * 20_000_000
n = 0..9
```

If a sample time is outside the SMPL range, it is clamped:

```text
sample_time = clamp(sample_time, S[0], S[-1])
```

This means the first selected RGB frame can clamp its first sample to the first
SMPL frame, and the last RGB frames can clamp future samples to the last SMPL
frame.

## Interpolation

Each 50 Hz sample time is evaluated on the original SMPL timeline.

For a sample time `t`, the script finds the neighboring SMPL source frames:

```text
j0 <= t <= j1
alpha = (t - S[j0]) / (S[j1] - S[j0])
```

Then it interpolates:

- `transl`: linear interpolation
- `betas`: linear interpolation or broadcast if constant
- `global_orient` and `body_pose`: axis-angle -> quaternion -> SLERP -> axis-angle

The interpolated SMPL windows are processed with the same Pico-style SMPL path
used by the existing offline SMPL exporter, then encoded with `model_encoder.onnx`
in SMPL mode.

## Output

The default output is written inside each sequence directory:

```text
<sequence-dir>/token/token.npz
```

Important fields:

- `token_state`: encoded tokens, shape `(selected_rgb_frames, 64)`
- `rgb_frame_indices`: original RGB frame indexes
- `rgb_relative_timestamps_ns`: RGB anchor timestamps for each token
- `smpl_sample_relative_timestamps_ns`: 10-frame 50 Hz SMPL window per token
- `smpl_unclamped_sample_relative_timestamps_ns`: pre-clamp window timestamps
- `smpl_sample_clamped_mask`: whether each SMPL sample was clamped
- `source_smpl_left_indices`
- `source_smpl_right_indices`
- `source_smpl_interp_alpha`
- `time_zero_ns`
- `time_zero_source`
- `sampling_policy = "rgb_anchor_50hz_future_window_slerp_clamp"`

These metadata fields make every token row traceable back to the RGB frame and
the SMPL source frames used for interpolation.

## Usage

Create the dedicated token-export Python environment once from the repository
root:

```bash
cd /path/to/GR00T-WholeBodyControl
bash install_scripts/install_token_export.sh
```

Activate that environment before exporting tokens:

```bash
source gear_sonic_deploy/scripts/setup_token_export_env.sh
```

This environment is intentionally separate from the deploy runtime environment.
It installs only the Python dependencies needed by this offline exporter, such
as `gear_sonic`, CPU-only PyTorch, `numpy`, `scipy`, `tqdm`, and
`onnxruntime`. The installer uses the PyTorch CPU wheel index and installs the
local `gear_sonic` package with `--no-deps` so dependency resolution does not
pull in a CUDA PyTorch build.

For a root containing many sequences:

```bash
python gear_sonic_deploy/reference/export_nymeria_rgb_smpl_tokens.py \
  /path/to/ny_batch_root \
  --overwrite
```

By default, the exporter shows two progress bars when `tqdm` is available:

- `Sequences`: one tick per discovered sequence
- `<sequence_id> RGB frames`: one frame-counted bar for token encoding within
  that sequence

To disable progress bars, for example in redirected logs:

```bash
python gear_sonic_deploy/reference/export_nymeria_rgb_smpl_tokens.py \
  /path/to/ny_batch_root \
  --overwrite \
  --no-progress
```

For one sequence, pass that sequence directory directly:

```bash
python gear_sonic_deploy/reference/export_nymeria_rgb_smpl_tokens.py \
  /path/to/ny_batch_root/<sequence_id> \
  --overwrite
```

For a quick smoke test on a few RGB frames:

```bash
python gear_sonic_deploy/reference/export_nymeria_rgb_smpl_tokens.py \
  /path/to/ny_batch_root \
  --max-rgb-frames 10 \
  --overwrite
```

Explicit paths are also supported:

```bash
python gear_sonic_deploy/reference/export_nymeria_rgb_smpl_tokens.py \
  --smpl-npz /path/to/nymeria_smpl.npz \
  --rgb-timestamps /path/to/head_video/timestamps.npz \
  --output /path/to/output_tokens.npz \
  --overwrite
```

## Notes

- One output token corresponds to one selected RGB frame.
- The internal SMPL encoder window is always 50 Hz.
- The RGB video frame rate does not control the SMPL encoder frame rate.
- Tail windows are intentionally clamped to the last SMPL frame.
- Use `.venv_token_export` for this offline exporter so it does not inherit the
  heavier deploy runtime setup.
