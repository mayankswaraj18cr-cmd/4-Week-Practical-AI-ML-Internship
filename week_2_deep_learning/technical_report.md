# Week 2 Technical Report

## Objective
Build reproducible PyTorch baselines for tabular prediction and image classification.

## Design
- `FeedForwardNetwork` isolates model architecture from data loading and optimization.
- `CNNImageClassifier` exposes `extract_features` so embeddings can be reused downstream.
- `ExperimentLogger` records optimizer and loss metadata in a portable JSON format.

## Reproducibility
Set `torch.manual_seed`, keep dataset splits fixed, and record Python, PyTorch, and CUDA versions with each run. Production training should add checkpointing, early stopping, and an external experiment tracker.
