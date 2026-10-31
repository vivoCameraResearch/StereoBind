<div align="center">
  <img src="assets/logo_stereobind.png" width="280" alt="StereoBind logo">

  <h1>Hear the World in Stereo</h1>
  <h3>Learning Dynamic Spatial Correspondence for Immersive Joint Video-Audio Generation</h3>

  <p><strong>Official repository for StereoBind</strong></p>

  <p>
    <a href="https://github.com/vivoCameraResearch/StereoBind"><img src="https://img.shields.io/badge/GitHub-StereoBind-181717?logo=github" alt="GitHub"></a>
    <a href="https://vivocameraresearch.github.io/Stereo-Bind-Project/"><img src="https://img.shields.io/badge/Project-Page-7c3aed?logo=githubpages&logoColor=white" alt="Project Page"></a>
    <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="Apache 2.0 License"></a>
    <a href="#getting-started"><img src="https://img.shields.io/badge/Status-Inference%20Guide%20Ready-16a34a" alt="Inference guide ready"></a>
    <a href="https://huggingface.co/MilanMagik/StereoBindCheckpoint"><img src="https://img.shields.io/badge/Checkpoints-Hugging%20Face-ff9d00?logo=huggingface" alt="Checkpoints"></a>
    <a href="https://huggingface.co/datasets/MilanMagik/StereoBindInferenceSamples"><img src="https://img.shields.io/badge/Inference%20Samples-Hugging%20Face-ff9d00?logo=huggingface" alt="Inference Samples"></a>
  </p>
</div>

> [!IMPORTANT]
> The installation and inference guide is available below. This repository currently contains the README and project logo; the `packages/` and `run_scripts/` directories are pending the inference-code release. The package-install and inference commands below require those directories to be uploaded first.

## TL;DR

**StereoBind** generates video and stereo audio jointly while keeping the perceived sound direction aligned with the visible source as it moves. It extends audio-visual generation from **what** happens and **when** it happens to **where it happens over time**.

## Release Plan

- [x] Paper and project page
- [x] Inference code and example prompts
- [x] Pretrained model checkpoints
- [ ] Training code and configuration files
- [ ] StereoWorld-29K data and construction pipeline
- [ ] StereoWorldBench evaluation code

## Getting Started

### Environment setup

The commands below use Linux / Bash and Python 3.10 or newer. Use a separate Python environment and an NVIDIA GPU with a driver compatible with the PyTorch CUDA 12.6 build. FFmpeg must be available on `PATH`: the default launcher saves both stages and uses FFmpeg to combine the Stage 2 video with the Stage 1 audio.

#### 1. Clone the repository

```bash
git clone https://github.com/vivoCameraResearch/StereoBind.git
cd StereoBind

# Ubuntu / Debian: install FFmpeg if it is not already available.
sudo apt-get update
sudo apt-get install -y ffmpeg

conda create --name stereobind python=3.10
conda activate stereobind
```

#### 2. Install dependencies

Run the following commands from the repository root to install PyTorch 2.9.0 with CUDA 12.6, followed by the two local packages:

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -e packages/ltx-core
pip install -e packages/ltx-pipelines
```

#### 3. Download the required models and checkpoints

Install the Hugging Face CLI and download the LTX-2.3 distilled model, spatial upscaler, Gemma text encoder, and StereoBind checkpoints. The examples below use local directories under the repository root. You can choose other locations, then update the paths in `run_scripts/inference.sh`.

Before downloading Gemma, accept the access conditions on the [Gemma model page](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) using your Hugging Face account, then authenticate locally with `hf auth login` as shown below.

```bash
python -m pip install huggingface_hub

# LTX-2.3 distilled model and spatial upscaler
hf download Lightricks/LTX-2.3 \
  ltx-2.3-22b-distilled-1.1.safetensors \
  ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --local-dir "./models/LTX-2.3"

# Gemma text encoder
hf auth login
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir "./models/gemma-3-12b-it-qat-q4_0-unquantized"

# StereoBind checkpoints
hf download MilanMagik/StereoBindCheckpoint \
  --local-dir "./models/StereoBindCheckpoint"
```

Download the entire Gemma repository, including the model shards, tokenizer, and configuration files. `GEMMA_ROOT` must point to this directory.

The checkpoint repository contains two variants:

| Variant | Checkpoint subdirectory | `SUMMARY_TOKEN_NUM` |
| --- | --- | --- |
| Token32 (launcher default) | `StereoBindCheckpointToken32` | `32` |
| Token64 | `StereoBindCheckpointToken64` | `64` |

Each variant contains the following three files:

```text
lora_weights_step_08000.safetensors
track_conditioning_step_08000.safetensors
simple_summary_step_08000.safetensors
```

Set `CHECKPOINT_DIR` to the variant subdirectory containing these files. Keep all three weights from the same variant and checkpoint step, and match `SUMMARY_TOKEN_NUM` to that variant.

#### 4. Download the inference samples

The samples are hosted in a Hugging Face **dataset** repository, so `--repo-type dataset` is required:

```bash
hf download MilanMagik/StereoBindInferenceSamples \
  --repo-type dataset \
  --local-dir "./inference_samples"
```

### Run inference

#### 1. Configure the model paths

Edit `run_scripts/inference.sh` from the inference-code release. The default model paths match the download locations above; replace them if you stored the files elsewhere.

| Variable | What to configure |
| --- | --- |
| `DISTILLED_CHECKPOINT_PATH` | Full path to `ltx-2.3-22b-distilled-1.1.safetensors`. |
| `SPATIAL_UPSAMPLER_PATH` | Full path to `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`. |
| `GEMMA_ROOT` | Directory containing the downloaded Gemma model, tokenizer, and configurations. |
| `CHECKPOINT_DIR` | Directory containing the three StereoBind checkpoint files, e.g. `models/StereoBindCheckpoint/StereoBindCheckpointToken32`. |
| `SUMMARY_TOKEN_NUM` | `32` or `64`, matching the selected checkpoint variant. (Default 32) |
| `CHECKPOINT_STEP` | Checkpoint filename suffix; the released files use `08000`. |

#### 2. Set BATCH_JSON

Set `BATCH_JSON` to the path of a JSON file from `MilanMagik/StereoBindInferenceSamples`:

```bash
# Replace your_samples.json with the actual downloaded JSON filename.
BATCH_JSON="${PROJECT_ROOT}/inference_samples/your_samples.json"
```

The batch loader accepts either a JSON array or an object with a `samples` array. Each sample requires `name`, an integer `seed`, `prompt`, `reference_video`, and `track` for the default StereoBind configuration. Example structure for your own inputs (replace the video and track paths with existing files):

```json
{
  "samples": [
    {
      "name": "demo",
      "seed": 42,
      "prompt": "A person walks from left to right while speaking.",
      "reference_video": "videos/demo.mp4",
      "track": "tracks/demo.pt"
    }
  ]
}
```

#### 3. Set OUTPUT_PATH and launch

Set the output **directory** in `run_scripts/inference.sh`:

```bash
OUTPUT_PATH="${PROJECT_ROOT}/outputs"
```

Then run:

```bash
bash run_scripts/inference.sh
```

The launcher invokes `ltx_pipelines.ic_lora_customize_batch_two_stage` with a target resolution of **1280 × 768**, **121 frames**, and **24 fps**. Stage 1 generates at **640 × 384**; Stage 2 upsamples and refines the video. Per-sample resolution, frame count, and frame rate fields in the JSON override the launcher defaults.

Results are written to a configuration-specific subdirectory under `OUTPUT_PATH`. The default `--save-stage-outputs` option saves:

- `generated/*_stage1.mp4`: Stage 1 output.
- `generated/*_stage2.mp4`: Stage 2 output.
- `generated/*_stage2_with_stage1_audio.mp4`: Stage 2 video paired with Stage 1 audio.
- `concat/`: Side-by-side reference and generated videos.

## Citation

If you find this project useful, please consider citing our work. The official BibTeX entry will be added when the paper is released.

> **Hear the World in Stereo: Learning Dynamic Spatial Correspondence for Immersive Joint Video-Audio Generation**

## Acknowledgements

StereoBind builds on [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2). Please refer to the upstream project for the base model implementation and its license.

## License

This project is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Contact

Questions and suggestions are welcome through [GitHub Issues](https://github.com/vivoCameraResearch/StereoBind/issues).
