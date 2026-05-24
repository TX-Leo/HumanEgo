<h1 align="center">
  <img src="assets/website/images/humanego_logo_trim.png" height="48" />
  HumanEgo
</h1>

<h3 align="center">
  Zero-Shot Robot Learning<br/>
  from Minutes of Human Egocentric Videos
</h3>

<p align="center">
  <a href="https://tx-leo.github.io">Zhi (Leo) Wang</a> &nbsp;·&nbsp;
  <a href="https://bottle101.github.io/">Botao He</a> &nbsp;·&nbsp;
  <a href="https://colinyu1.github.io/">Kelin Yu</a> &nbsp;·&nbsp;
  <a href="https://sjlee.cc/">Seungjae Lee</a> &nbsp;·&nbsp;
  <a href="https://ruohangao.github.io/">Ruohan Gao</a> &nbsp;·&nbsp;
  <a href="https://furong-huang.com/">Furong Huang</a> &nbsp;·&nbsp;
  <a href="https://robotics.umd.edu/clark/faculty/350/Yiannis-Aloimonos">Yiannis Aloimonos</a>
</p>

<p align="center"><b>University of Maryland</b></p>

<p align="center">
  <a href="https://humanego-ai.github.io"><b>Project Page</b></a>
</p>

<p align="center">
  <img src="assets/teaser.gif" alt="HumanEgo teaser" width="100%" />
</p>

---

## Installation

### One-click setup

```bash
git clone https://github.com/TX-Leo/HumanEgo.git
cd HumanEgo
conda create -n humanego python=3.11 -y
conda activate humanego
bash setup.sh
```

This creates a conda environment `humanego` (Python 3.11) and installs everything automatically, including:

- Core ML stack (PyTorch + CUDA, transformers, SAM2, …)
- Orient-Anything V2 (auto-installed from GitHub as a pip package)
- CoTracker (auto-installed from GitHub)
- Hand tracking methods (MediaPipe, WiLoR, HaMeR)
- Robot hardware SDKs (RealSense, Trossen Arm)

### Optional flags

```bash
SKIP_HAND=1     bash setup.sh   # skip hand-tracking packages (MediaPipe, WiLoR, HaMeR)
SKIP_HARDWARE=1 bash setup.sh   # skip pyrealsense2 & trossen-arm (non-robot machines)
PREDOWNLOAD=1   bash setup.sh   # pre-download all model weights up front
```

---

## Data Collection

> **TODO** — documentation coming soon.

---

## Preprocess

> **TODO** — documentation coming soon.

---

## Training

> **TODO** — documentation coming soon.

---

## Inference

> **TODO** — documentation coming soon.

---

## Acknowledgements

This project builds on excellent open-source work, including
[Project Aria](https://www.projectaria.com/) (Gen 1 glasses &amp;
[MPS](https://facebookresearch.github.io/projectaria_tools/docs/intro)),
[Trossen Arm](https://www.trossenrobotics.com/),
[CoTracker3](https://github.com/facebookresearch/co-tracker),
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO),
[SAM 2](https://github.com/facebookresearch/sam2),
[HaMeR](https://github.com/geopavlakos/hamer),
[WiLoR](https://github.com/rolpotamias/WiLoR),
[MediaPipe](https://github.com/google-ai-edge/mediapipe),
[LaMa](https://github.com/advimman/lama),
and [Orient-Anything](https://github.com/SpatialVision/Orient-Anything).
