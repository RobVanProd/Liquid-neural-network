# Liquid-neural-network

This repo contains several PyTorch liquid-network experiments, but the current test suite did not collect on 2026-05-28 because required dependencies and modules were missing.

## What It Is

`Liquid-neural-network` is a neural-architecture experiment repo. It contains a baseline liquid neural network, larger variants with residual and attention modules, a multi-scale `SuperLiquidNetwork`, Streamlit UI code, synthetic dataset generators, benchmark scaffolding, and tests/examples for time-series style modeling.

The interesting part is that the repo explores liquid dynamics alongside attention, multi-scale layers, and evolution-style adaptation hooks in one playground.

## Current Status

`python -m pytest -q` did not pass in the documentation environment. Collection failed because `sklearn`, `seaborn`, and a referenced `liquid_s4` module were missing.

That means this README should not claim accuracy, benchmark performance, or working end-to-end training. The codebase is best described as an experimental collection of model variants.

## Tech Stack

- Python
- PyTorch
- NumPy
- scikit-learn dependency
- Streamlit
- Plotly/matplotlib/seaborn dependencies
- PyQt6 dependency

## Limitations

Dependencies need to be installed and the missing `liquid_s4` reference needs to be resolved before tests can validate behavior. The README should not call the system self-improving unless a specific experiment log backs that claim.
