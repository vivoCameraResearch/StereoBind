<div align="center">
  <img src="assets/logo_stereobind.png" width="280" alt="StereoBind logo">

  <h1>Hear the World in Stereo</h1>
  <h3>Learning Dynamic Spatial Correspondence for Immersive Joint Video-Audio Generation</h3>

  <p><strong>Official repository for StereoBind</strong></p>

  <p>
    <a href="https://github.com/vivoCameraResearch/StereoBind"><img src="https://img.shields.io/badge/GitHub-StereoBind-181717?logo=github" alt="GitHub"></a>
    <a href="https://vivocameraresearch.github.io/Stereo-Bind-Project/"><img src="https://img.shields.io/badge/Project-Page-7c3aed?logo=githubpages&logoColor=white" alt="Project Page"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="Apache 2.0 License"></a>
    <img src="https://img.shields.io/badge/Status-Coming%20Soon-f59e0b" alt="Coming soon">
  </p>
</div>

> [!IMPORTANT]
> This repository has been initialized ahead of the public release. The paper, code, checkpoints, data, and evaluation suite will be made available here.

## TL;DR

**StereoBind** generates video and stereo audio jointly while keeping the perceived sound direction aligned with the visible source as it moves. It extends audio-visual generation from **what** happens and **when** it happens to **where it happens over time**.

## Overview

Joint video-audio models can synthesize plausible visuals and synchronized sounds, but temporal synchronization alone does not guarantee an immersive result. A sound-producing object may move across the frame while its generated audio remains spatially static or shifts inconsistently.

We formulate this missing capability as **Dynamic Spatial Correspondence**: the evolving spatial behavior of generated stereo audio should agree with the motion of its visible source. StereoBind uses a visual source trajectory as a compact geometric interface shared by video motion and audio generation.

| Component | Purpose |
| --- | --- |
| **StereoWorld-29K** | A stereo audio-visual dataset with continuous, geometry-grounded spatial supervision from curated real videos and trajectory-controlled synthetic scenes. |
| **StereoBind** | A trajectory-conditioned framework for immersive joint video-audio generation. |
| **StereoWorldBench** | An evaluation suite for measuring whether generated audio follows the visible sound source over time. |

## Method

StereoBind introduces trajectory information into a pretrained joint audio-video generator through three complementary components:

- **Simple Summary Tokens** build a compact video-to-audio pathway, distilling source-related visual content and motion without transferring the full visual token sequence.
- **Spatial Track Encoder (STE)** converts normalized 2D source trajectories into temporally aligned conditions for the audio stream.
- **ResidualTrackRoPE** augments temporal attention with relative trajectory changes, enabling the model to reason about how spatial relationships evolve over time.

Together, these components connect visual content, absolute source position, and relative source motion to stereo audio generation.

## StereoWorld-29K

StereoWorld-29K is designed to provide the supervision required for Dynamic Spatial Correspondence. Its construction pipeline combines:

1. quality- and motion-aware audio-visual data curation;
2. sound-source identification and tracking;
3. sequence-consistent geometry and camera-motion recovery; and
4. time-varying stereo rendering with dynamic room acoustics.

The resulting samples preserve the semantics and timing of audio-visual events while adding continuous spatial cues that follow source and listener motion.

## Release Plan

- [ ] Paper and project page
- [ ] Inference code and example prompts
- [ ] Pretrained model checkpoints
- [ ] Training code and configuration files
- [ ] StereoWorld-29K data and construction pipeline
- [ ] StereoWorldBench evaluation code

## Getting Started

Installation and inference instructions will be added with the first code release. Please **watch** this repository or check the release plan above for updates.

## Citation

If you find this project useful, please consider citing our work. The official BibTeX entry will be added when the paper is released.

> **Hear the World in Stereo: Learning Dynamic Spatial Correspondence for Immersive Joint Video-Audio Generation**

## License

This project is released under the [Apache License 2.0](LICENSE).

## Contact

Questions and suggestions are welcome through [GitHub Issues](https://github.com/vivoCameraResearch/StereoBind/issues).
