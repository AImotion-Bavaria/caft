# CAFT: Constraint-Guided Adaptive Fine-Tuning of Neural Networks

## Project Overview

CAFT is a research codebase accompanying the paper:

> **Making Limited Data Count: Constraint-Guided Adaptive Fine-Tuning of Neural Networks for Cost-Aware Defect Classification**  
> Kristina Dachtler, Alexander Schiendorfer  
> ETFA 2026 — Special Session: Addressing Data Scarcity

In manufacturing environments, labeled data is often scarce, imbalanced, and of uncertain quality. Under these conditions, optimizing for conventional aggregate metrics quickly reaches a plateau — and further improvements do not necessarily translate into better operational outcomes. Defining a suitable cost matrix for cost-sensitive learning is non-trivial, and static cost-sensitive training methods tend to collapse on small, imbalanced datasets.

**CAFT** addresses these challenges with a two-phase training strategy:

1. **Phase 1 — Pre-Training:** A neural network is trained with standard Cross-Entropy (or Weighted Cross-Entropy) loss to establish stable feature representations.
2. **Phase 2 — Constraint-Aware Fine-Tuning:** Starting from the pre-trained weights, misclassification penalties are incrementally increased until predefined operational decision constraints are satisfied. The method does not require a predefined cost matrix — instead, constraint-relevant cost entries are built up adaptively via Lagrangian multipliers.

### Key Results

- CAFT achieves **higher constraint satisfaction rates** than static cost-sensitive methods across two datasets, without model collapse.
- CAFT maintains the **highest F1-score among all methods that fulfill the decision constraint**.
- Compared to post-hoc threshold tuning, CAFT shows **more robust generalization** from validation to test data.
- The approach **scales naturally to multiple simultaneous constraints**, a setting where threshold-based alternatives become impractical.

### Datasets

- **Paint Defects (real-world):** Automotive paint defect classification with 2,681 samples, 5 features, and 3 classes (not included due to confidentiality reasons).
- **Steel Plates Faults (public benchmark):** Adapted from the [UCI Steel Plates Faults dataset](https://doi.org/10.24432/C52S3Z) with 1,941 samples, 27 features, grouped into 3 classes.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AImotion-Bavaria/caft
   cd CAFT
   ```
2. Install dependencies (Python >=3.10):
   ```bash
   pip install -e .
   ```
   This uses the dependencies specified in `pyproject.toml`.

## Directory Structure (excerpt)

- `src/` — Core modules (models, training, evaluation, utilities)
- `config/` — Central configuration (paths)
- `artifacts/` — Data, models (model_weights from pre-training), and results (auto-generated)
- `experiments/` — Experiment scripts for different datasets
- `create_experiment_plots.py` — Aggregates and plots experiment results
- `create_experiment_summary.py` — Summarizes results from text outputs

## Running an Example Experiment

To run a neural network experiment on the steel plates dataset:
```bash
python experiments/steel_plates/01_torch_NN_training_SP.py
```

## Aggregating Results

Results are aggregated and plotted after running experiments:
```bash
python create_experiment_plots.py
python create_experiment_summary.py
```

<!-- ## Citation

```bibtex
@inproceedings{dachtler2026caft,
  title={Making Limited Data Count: Constraint-Guided Adaptive Fine-Tuning of Neural Networks for Cost-Aware Defect Classification},
  author={Dachtler, Kristina and Schiendorfer, Alexander},
  booktitle={IEEE International Conference on Emerging Technologies and Factory Automation (ETFA)},
  year={2026}
}
``` -->

## License

This project is licensed under the MIT License.

## Author

Kristina Dachtler ([kristina.dachtler@thi.de](mailto:kristina.dachtler@thi.de))
