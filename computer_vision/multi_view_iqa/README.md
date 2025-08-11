# Multi-view MOS Prediction

This folder contains a lightweight implementation of a multi-view
stereoscopic image quality assessment (IQA) model.  The code follows the
architecture proposed in recent IQA literature, combining a **Distortion
Feature Encoder (DFE)** with a **Multi-view Interaction Aggregation (MIA)**
transformer to predict Mean Opinion Score (MOS) values for six input
views.

The implementation is intentionally compact to serve as a starting point
for further research.

## Components

* `model.py` – definition of `DFEModule`, `MIAModule` and the combined
  `MultiViewIQAModel`.
* `dataset.py` – `MVSDataset` for loading six-view images and optional
  depth maps from a metadata file.
* `losses.py` – a collection of loss functions described in the design
  document (Smooth L1, ranking, z-score MSE and anchor-based losses).
* `utils.py` – helper utilities including a five-parameter logistic
  mapping used during evaluation.
* `train.py` – minimal example training loop.

Each file contains extensive docstrings explaining the input/output
shapes of the involved tensors.

## Usage

Prepare a metadata text file where each line contains the six image
paths, optional six depth map paths and the MOS value.  Then run:

```bash
python -m computer_vision.multi_view_iqa.train path/to/meta.txt
```

The training script is deliberately minimal and intended for small scale
experiments or as a template to build upon.
