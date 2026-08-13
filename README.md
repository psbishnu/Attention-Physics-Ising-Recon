# HELP — Attention-Guided Physics-Regularized Inverse Ising Reconstruction

This file provides practical instructions for running the Python programs included with this repository. The codes cover dataset generation, the main deep-learning reconstruction model, evaluation, reference-method comparison, ablation studies, robustness tests, physics-loss analysis, and quantitative attention interpretation.

## 1. Code files

| File | Purpose |
|---|---|
| `MCD.py` | Generates clean and corrupted 2D Ising datasets using Wolff cluster updates. |
| `DL.py` | Main physics-informed inverse reconstruction model. It reconstructs corrupted spins and jointly predicts temperature and phase. |
| `Evaluation.py` | Evaluates saved temperature and phase predictions and generates publication-ready plots and summary metrics. |
| `Comparison.py` | Compares the proposed method with conventional imputation baselines such as Mean and KNN imputation. |
| `Ablation2.py` | Architectural/component ablation study, including attention removal, simpler CNNs, mask removal, loss-weight changes, and corruption-level experiments. |
| `Ablation3.py` | Optimization and physics-loss analysis: reconstruction-only versus full physics-guided learning and physics-weight sensitivity. |
| `ablation5.py` | Individual physics-loss ablation: Full, No Energy, No Magnetization, No Correlation, No Binary, and scaled physics losses. |
| `Ablation6.py` | Robustness analysis under different spin-flip noise and random-masking levels. |
| `Ablation7.py` | Quantitative interpretation of learned spatial attention using domain boundaries, local disagreement, AUROC, shuffled controls, and temperature regimes. |

> GitHub filenames may be renamed from the uploaded names (for example, `DL(2).py` → `DL.py`) for clarity. If the original names are retained, use those exact filenames in the commands below.

## 2. Main methodology

The main model receives a corrupted Ising lattice and its observation mask. It performs three related tasks:

1. reconstruction of the clean spin configuration;
2. temperature regression; and
3. ferromagnetic/paramagnetic phase classification.

The composite objective contains reconstruction, energy, absolute-magnetization, second-neighbor correlation (`C2`), binary-spin, temperature, and phase terms.

**Important:** `C1` is retained as an evaluation observable but is not treated as an independent physics loss in the updated codes because nearest-neighbor `C1` is directly related to the Ising energy for the adopted nearest-neighbor Hamiltonian. `C2` is therefore used as the independent longer-range/second-neighbor correlation regularizer.

Typical loss coefficients used in the repository are:

```text
lambda_rec   = 2.0
lambda_E     = 0.1
lambda_M     = 0.1
lambda_C2    = 0.1
lambda_bin   = 0.5
lambda_T     = 1.0
lambda_phase = 1.0
```

The scripts use a fixed random seed of `123` unless explicitly changed.

## 3. Python environment

Python 3.10 or a compatible recent Python 3 version is recommended.

Create an environment if required:

```bash
conda create -n ising-recon python=3.10 -y
conda activate ising-recon
```

Install the principal dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tqdm torch
```

A CUDA-enabled PyTorch installation is recommended for model training. Follow the official PyTorch installation instructions appropriate to the CUDA version installed on your system/HPC cluster.

## 4. Dataset generation

Run:

```bash
python MCD.py
```

The generator uses the Wolff cluster algorithm. Its default configuration contains 50 temperature points over `T = 1.0–4.0`, 50,000 configurations, and 1,000 configurations per temperature. The default thermalization and sampling gaps are 2,000 and 200 Wolff updates, respectively.

To generate another lattice size, edit:

```python
L = 32
```

and set it to `64` or `128`.

The generator creates clean and corrupted CSV datasets of the form:

```text
MCD32.csv
MCDN32.csv
MCD64.csv
MCDN64.csv
MCD128.csv
MCDN128.csv
```

The default corruption parameters in the generator are a spin-flip probability of `0.10` and masking probability of `0.30`.

## 5. Recommended directory organization

The supplied programs contain several relative paths. A convenient HPC organization is:

```text
project/
├── JOB5_Noise/
│   └── J5Data/
│       ├── MCD32.csv
│       ├── MCDN32.csv
│       ├── MCD64.csv
│       ├── MCDN64.csv
│       ├── MCD128.csv
│       └── MCDN128.csv
│
└── JOB_SCRIPTA/
    ├── DL.py
    ├── Evaluation.py
    ├── Comparison.py
    ├── Ablation2.py
    ├── Ablation3.py
    ├── ablation5.py
    ├── Ablation6.py
    ├── Ablation7.py
    └── HELP.md
```

Some scripts use slightly different relative paths. **Check the user-configuration section at the beginning of every script before running it.**

## 6. Main model

Run:

```bash
python DL.py
```

Before execution, set the required lattice size and corresponding paths, for example:

```python
L = 32
CLEAN_CSV = Path("../JOB5_Noise/J5Data/MCD32.csv")
NOISY_CSV = Path("../JOB5_Noise/J5Data/MCDN32.csv")
OUT_DIR = "./Generated_L128"
```

The main implementation uses Adam/Adam-style optimization settings with a learning rate of `3e-4`, early stopping, and the composite physics-guided objective. The current uploaded `DL.py` configuration uses 40 maximum epochs and patience 7.

Typical outputs include:

```text
best_model.pth
metrics_global.csv
metrics_per_temp_bin.csv
preds_test.csv
training_times.csv
physics_plots.pdf
training_plots.pdf
configs_bw.pdf
accuracy_vs_temperature.pdf
```

Run the model independently for `L = 32`, `64`, and `128` as required.

## 7. Evaluation

`Evaluation.py` expects a prediction CSV containing:

```text
T_true
T_pred
Phase_true
Phase_pred
```

Set:

```python
L = 128
```

and ensure the corresponding prediction file is in the working directory. Then run:

```bash
python Evaluation.py
```

The program calculates temperature MAE and macro phase-classification statistics and produces prediction/confusion-matrix PDFs and an overall summary CSV.

## 8. Reference-method comparison

Run:

```bash
python Comparison.py
```

Edit:

```python
L = 32
```

to select `32`, `64`, or `128`.

The script compares the reconstruction with conventional Mean and KNN imputation baselines and computes global and temperature-resolved reconstruction/physics statistics. Confirm that `CLEAN_CSV`, `NOISY_CSV`, and `PROPOSED_OUT` point to the correct data and proposed-model output directory.

## 9. Architectural/component ablation

Run:

```bash
python Ablation2.py
```

The script investigates requested architectural and component variants, including:

- attention removal;
- simplified encoder/decoder;
- vanilla CNN;
- individual physics-loss variants;
- loss-weight tuning;
- mask-channel removal;
- corruption/missing-level variation; and
- reduced channel widths.

Set the lattice size and dataset paths in the user-parameter section before execution.

## 10. Optimization and physics-loss analysis

Run:

```bash
python Ablation3.py
```

This code performs the compact reviewer-oriented analysis:

- reconstruction-only versus full physics-guided optimization;
- convergence analysis; and
- physics-loss coefficient sensitivity.

The current script uses:

```text
Maximum epochs: 15
Learning rate: 3e-4
Early-stopping patience: 3
Physics coefficients tested: 0.05, 0.10, 0.20
```

It supports lattice-size selection through the environment variable:

```bash
ISING_L=32 python Ablation3.py
ISING_L=64 python Ablation3.py
ISING_L=128 python Ablation3.py
```

The lattice-dependent batch sizes in this script are `64`, `16`, and `4` for `L=32`, `64`, and `128`, respectively.

## 11. Individual physics-loss ablation

Run:

```bash
python ablation5.py
```

Set:

```python
DATASET_NAME = "MCD32"
```

or `MCD64` / `MCD128`.

The experiments include:

```text
Full
No_Energy
No_Magnetization
No_Correlation
No_Binary
Physics_x0.5
Physics_x2.0
```

Here, `No_Correlation` removes the **C2** loss. `C1` remains an evaluation quantity.

## 12. Robustness analysis

Run:

```bash
python Ablation6.py
```

Select:

```python
DATASET_NAME = "MCD32"
```

or another lattice size.

The default robustness protocol trains at 30% spin-flip noise and evaluates at 10%, 30%, and 50%. It separately trains at 40% random masking and evaluates at 20%, 40%, and 60% masking.

The script automatically falls back to CPU when CUDA is unavailable, although GPU execution is preferable.

## 13. Quantitative attention analysis

Run:

```bash
python Ablation7.py
```

Change only:

```python
L = 32
```

to `64` or `128` when appropriate.

This analysis **does not retrain the model**. It loads an existing `best_model.pth`, reconstructs the deterministic test split, extracts spatial-attention maps, and quantifies their relationship with domain boundaries and local spin disagreement.

The supplied HPC checkpoint convention is:

```python
CHECKPOINT_OVERRIDE = f"/home/partha02965/JOB_SCRIPTA/Generated_L{L}/best_model.pth"
```

Modify this path if your repository or checkpoint directory differs.

Principal outputs include:

```text
attention_per_sample_L{L}.csv
attention_summary_regimes_L{L}.csv
attention_summary_temperature_L{L}.csv
attention_physics_curves_L{L}.pdf
attention_examples_L{L}.pdf
attention_table_L{L}.tex
```

The shuffled-attention AUROC serves as a random-control reference.

## 14. Running on an HPC system

Activate the environment first:

```bash
conda activate deeplearn
```

Then execute a script directly on an allocated compute node, for example:

```bash
python DL.py
```

For PBS jobs, the Python command can be placed in the submitted job script.

Check GPU availability with:

```bash
nvidia-smi
```

If `nvidia-smi` is unavailable and no GPU has been allocated, CUDA-based scripts may fall back to CPU where this behavior is implemented. Training large `L=128` models on CPU can be substantially slower.

## 15. Important checks before each run

Before executing a program, verify:

1. the selected lattice size (`L` or `DATASET_NAME`);
2. clean/noisy CSV paths;
3. output directory;
4. checkpoint path for attention analysis;
5. CUDA/GPU availability;
6. batch size and memory requirements; and
7. that outputs from earlier stages required by evaluation/comparison scripts already exist.

Do not mix checkpoints or prediction files generated for different lattice sizes.

## 16. Reproducibility

For reproducibility, preserve:

- the random seed;
- train/validation/test splitting procedure;
- lattice size;
- corruption parameters;
- loss coefficients;
- optimizer and learning rate;
- batch size;
- early-stopping settings; and
- exact dataset version.

Generated CSV results, model checkpoints, and run metadata should be retained together with the code version used to produce them.

## 17. Suggested execution order

A typical complete workflow is:

```text
MCD.py
   ↓
DL.py
   ↓
Evaluation.py
   ↓
Comparison.py
   ↓
Ablation2.py / Ablation3.py / ablation5.py
   ↓
Ablation6.py
   ↓
Ablation7.py
```

Dataset generation is normally performed once per lattice size. The trained checkpoint from the main model is required by analyses that explicitly load the saved model.

## 18. Citation and research use

If this repository accompanies a manuscript, please cite the associated publication once bibliographic information is available. When reproducing reported results, use the same datasets, lattice sizes, random seed, corruption protocol, and parameter settings described in the manuscript and code.

## 19. Notes

- The repository targets 2D Ising lattices with periodic boundary conditions.
- `Tc = 2.269` is used as the reference critical temperature in the analysis codes.
- `C2` is the independent correlation regularizer in the updated physics-guided training codes.
- `C1` may still appear in output plots/tables because it remains useful as a physical evaluation observable.
- Large lattice sizes require substantially more memory and computation.
- Always inspect the configuration block near the top of each script before launching a long HPC job.
