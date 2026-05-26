<p align="center">
  <a href="https://humanego-ai.github.io">
    <img src="assets/title/hero.png" alt="HumanEgo — Zero-Shot Robot Learning from Minutes of Human Egocentric Videos" width="100%" />
  </a>
</p>

<p align="center">
  <a href="https://tx-leo.github.io">Zhi (Leo) Wang</a> &nbsp;·&nbsp;
  <a href="https://bottle101.github.io/">Botao He</a> &nbsp;·&nbsp;
  <a href="https://colinyu1.github.io/">Kelin Yu</a> &nbsp;·&nbsp;
  <a href="https://sjlee.cc/">Seungjae Lee</a> &nbsp;·&nbsp;
  <a href="https://ruohangao.github.io/">Ruohan Gao</a> &nbsp;·&nbsp;
  <a href="https://furong-huang.com/">Furong Huang</a> &nbsp;·&nbsp;
  <a href="https://robotics.umd.edu/clark/faculty/350/Yiannis-Aloimonos">Yiannis Aloimonos</a>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2605.24934"><img src="assets/title/btn_paper.png" alt="Paper" height="60" /></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2605.24934"><img src="assets/title/btn_arxiv.png" alt="arXiv" height="60" /></a>
  &nbsp;
  <a href="https://github.com/TX-Leo/HumanEgo"><img src="assets/title/btn_code.png" alt="Code" height="60" /></a>
  &nbsp;
  <a href="https://youtu.be/pdL46diijuY"><img src="assets/title/btn_video.png" alt="Video" height="60" /></a>
  &nbsp;
  <a href="#bibtex"><img src="assets/title/btn_bibtex.png" alt="BibTeX" height="60" /></a>
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

See [`datacollection/README_data_collection.md`](datacollection/README_data_collection.md)
for the end-to-end guide on recording your own Project Aria data and running
MPS (SLAM + hand tracking) on it.

---

## Preprocess

<p align="center">
  <img src="assets/data_collection.gif" alt="HumanEgo preprocessing visualization" width="100%" />
</p>

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

---

<h2 id="bibtex">BibTeX</h2>

```bibtex
@misc{humanego2026,
  title         = {HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos},
  author        = {Wang, Zhi and He, Botao and Yu, Kelin and Lee, Seungjae and Gao, Ruohan and Huang, Furong and Aloimonos, Yiannis},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```
