# Hybrid ML-Physics Climate Emulation (ClimSim)

A PyTorch Lightning pipeline for training, evaluating, and experimenting with neural network parameterizations of sub-grid atmospheric physics (clouds, convection, and radiative transfer) on the **ClimSim** dataset (Yu, Hu et al., 2024).

## Overview

Traditional General Circulation Models (GCMs) struggle to resolve fine-scale cloud and convective processes due to computational limits. This repository trains column-wise deep learning emulators to predict atmospheric tendencies and surface radiative/precipitation fluxes from large-scale thermodynamic state variables, featuring:

- **End-to-End Zarr Data Pipeline:** Chunked streaming download and preprocessing from Hugging Face into standardized Zarr v3 stores (`~4 GB` core variables or `~12 GB` full variable set).
- **Config-Driven PyTorch Lightning Training:** Reproducible training and evaluation with automatic hardware acceleration (CUDA, Apple Silicon MPS, CPU), mixed precision (`bfloat16`), and structured artifact logging.
- **Physical & Competition Evaluation:** Automated calculation of Kaggle global $R^2$ scores, column-wise $R^2$, area-weighted energy flux metrics ($\text{MAE}$, $\text{MSE}$ in $\text{W/m}^2$), and probabilistic metrics ($\text{CRPS}$, PIT histograms).

---

## Setup

### Using `uv` (Recommended)

```bash
uv sync
```

### Using Conda

```bash
conda env create -f environment.yaml
conda activate dlwp
```

---

## Quickstart Workflow

```bash
# 1. Download and preprocess ClimSim data into Zarr stores (~4 GB on disk)
uv run python fetch_data.py

# 2. Train and evaluate the baseline MLP
uv run python experiment.py --config configs/paper_mlp.yaml

# 3. Re-evaluate stored test predictions independently
uv run python score.py runs/paper_mlp/seed0/predictions.zarr
```

_(If using Conda instead of `uv`, omit `uv run` from the commands above.)_

### Pipeline Details

1. **Data Ingestion (`fetch_data.py`):** Streams raw simulation files from Hugging Face in blocks, extracts the target variables into `data/train.zarr` (`3,034,368` column-timesteps) and `data/test.zarr` (`1,681,920` column-timesteps), and immediately cleans up raw intermediate files. Pass `--full` to include all atmospheric variables (`~12 GB`).
2. **Training & Validation (`experiment.py`):** Trains the configured architecture using PyTorch Lightning and writes checkpoints, metric logs, and `scores.zarr` to `runs/<name>/seed<k>/`. Set `trainer.store_predictions: true` in the YAML config to export full test-set predictions to `predictions.zarr`.
3. **Evaluation (`score.py`):** Computes physical energy-flux metrics and Kaggle $R^2$ scores from any saved `predictions.zarr`. Pass `--probabilistic` to compute CRPS and PIT diagnostics for ensemble or distributional models.

### Interactive Exploration

`tutorial.ipynb` demonstrates the complete workflow on a lightweight subset (`~2 GB`) with data inspection and vertical-profile visualizations:

```bash
uv run jupyter lab tutorial.ipynb
```

See `GLOSSARY.md` for definitions of the atmospheric physics terms and variable names.

---

## Repository Structure

| File                        | Description                                                                                                               |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| `fetch_data.py`             | Streams raw ClimSim files from Hugging Face and builds `data/train.zarr` and `data/test.zarr`.                            |
| `experiment.py`             | Main CLI entry point for training, validation, testing, and logging runs.                                                 |
| `score.py`                  | Standalone evaluation module for deterministic ($R^2$, MAE, MSE in $\text{W/m}^2$) and probabilistic (CRPS, PIT) metrics. |
| `configs/paper_mlp.yaml`    | Hyperparameters and training schedule for the official ClimSim baseline MLP.                                              |
| `configs/tutorial.yaml`     | Lightweight configuration for rapid prototyping and verification.                                                         |
| `utils/dataset.py`          | `ClimSimData` PyTorch dataset handling feature standardization and inverse transforms back to `xarray`.                   |
| `utils/paper_mlp.py`        | Baseline multi-layer perceptron with physical non-negativity output clamping.                                             |
| `utils/lightning_module.py` | PyTorch Lightning module encapsulating training/validation/test steps, optimizers, and dataloaders.                       |
| `utils/config.py`           | Typed dataclass schema for experiment YAML configurations.                                                                |
| `tutorial.ipynb`            | End-to-end walkthrough notebook with exploratory data analysis and score visualizations.                                  |
| `GLOSSARY.md`               | Reference guide to atmospheric variables and physical units.                                                              |

---

## Dataset Schema

Each Zarr store is indexed across `(time, ncol, level)` with `384` horizontal grid columns and `60` vertical atmospheric levels. Each training sample represents a single atmospheric column at a single timestep, mapping **`124` input features** to **`128` target outputs** (`ptend_t`, `ptend_q0001`, and 8 surface radiative/precipitation fluxes).

```python
import xarray as xr
from utils.dataset import ClimSimData

# Inspect raw physical variables with xarray
ds = xr.open_zarr("data/train.zarr")
print(ds["state_t"])  # Air temperature (time, ncol, level)

# Load standardized PyTorch tensors for a specific time slice
train_data = ClimSimData("data/train.zarr", time=slice("0001-02", "0005-12"))
```

---

## Evaluation Metrics

- **Kaggle Score:** Mean $R^2$ across all `128` target channels over the test set (with zero-weighted stratospheric moisture levels scored as `1.0`).
- **Physical Flux Metrics:** Area-weighted $\text{MAE}$, $\text{MSE}$, and column-wise $R^2$ converted to physical energy flux units ($\text{W/m}^2$) following Yu, Hu et al. (2024).
- **Probabilistic Diagnostics:** Continuous Ranked Probability Score ($\text{CRPS}$) and Probability Integral Transform ($\text{PIT}$) histograms via `python score.py <predictions.zarr> --probabilistic`.

---

## References & Further Reading

### Core ClimSim Literature

- Yu, Hu et al. (2024). _ClimSim-Online: A Large Multi-scale Dataset and Framework for Hybrid ML-physics Climate Emulation._ [arXiv:2306.08754](https://arxiv.org/abs/2306.08754)
- Lin et al. (2025). _Crowdsourcing the Frontier: Advancing Hybrid Physics-ML Climate Simulation via a $50,000 Kaggle Competition._ [arXiv:2511.20963](https://arxiv.org/abs/2511.20963)

### ML for Weather & Climate Modeling

- Rasp, Pritchard & Gentine (2018). _Deep learning to represent sub-grid processes in climate models._ [arXiv:1806.04731](https://arxiv.org/abs/1806.04731)
- Kochkov et al. (2024). _Neural general circulation models for weather and climate (NeuralGCM)._ Nature. [arXiv:2311.07222](https://arxiv.org/abs/2311.07222) | [doi:10.1038/s41586-024-07744-y](https://doi.org/10.1038/s41586-024-07744-y)

### Architectures, Physical Constraints & Uncertainty

- Beucler et al. (2021). _Enforcing Analytic Constraints in Neural Networks Emulating Physical Systems._ [arXiv:1909.00912](https://arxiv.org/abs/1909.00912)
- Hu et al. (2024). _Stable machine-learning parameterization of subgrid processes with real geography and full-physics emulation._ [arXiv:2407.00124](https://arxiv.org/abs/2407.00124)
- Cachay et al. (2021). _ClimART: A Benchmark Dataset for Emulating Atmospheric Radiative Transfer in Weather and Climate Models._ [arXiv:2111.14671](https://arxiv.org/abs/2111.14671)
- Behrens et al. (2022). _Non-Linear Dimensionality Reduction With a Variational Encoder Decoder to Understand Convective Processes in Climate Models._ [arXiv:2204.08708](https://arxiv.org/abs/2204.08708)
- Lakshminarayanan, Pritzel & Blundell (2017). _Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles._ [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)
- Alet et al. (2025). _Skillful joint probabilistic weather forecasting from marginals (FGN)._ [arXiv:2506.10772](https://arxiv.org/abs/2506.10772)
