#  Attention-Guided Physics-Informed Deep Reconstruction of Noisy 2D Ising Spin Configurations

---

## Overview

This repository implements an advanced **physics-informed deep learning framework** for reconstructing clean spin configurations of the **two-dimensional Ising model** from noisy or partially observed data.

Unlike conventional denoising models, this framework integrates **thermodynamic constraints directly into training**, ensuring physically consistent reconstruction.

---

## Objectives

- Recover clean spin configurations from noisy inputs  
- Preserve thermodynamic observables (Energy, Magnetization, Correlation)  
- Perform **inverse inference** in lattice systems  
- Ensure consistency across ordered, critical, and disordered phases  

---

## Model Architecture

### 🔹 Input
- Noisy spin lattice (Y)
- Observation mask (M)

### 🔹 Network
- CNN Encoder with residual connections  
- Spatial Attention Module  
- CNN Decoder  

### 🔹 Outputs
- Reconstructed spins (Ŝ)  
- Temperature prediction (T̂)  
- Phase classification  

---

## Physics-Informed Loss

The total loss combines:

- Reconstruction Loss (observed + missing regions)
- Energy Loss (E consistency)
- Magnetization Loss (|M|)
- Correlation Loss (C1)
- Binary Spin Regularization
- Temperature Regression Loss
- Phase Classification Loss

---

## Dataset Format

CSV format:

Temperature, Phase, spin_0, spin_1, ..., spin_(L×L-1)

Supported naming:
- spin_0 ... spin_n  
- Spin1 ... SpinN  

Files required:
- MCD{L}.csv → Clean dataset  
- MCDN{L}.csv → Noisy dataset  

---

## Installation

```bash
pip install numpy pandas torch matplotlib tqdm scipy
```

---

## Usage

### Step 1: Configure parameters
Edit:

```python
L = 32
CLEAN_CSV = "./J5Data/MCD32.csv"
NOISY_CSV = "./J5Data/MCDN32.csv"
OUT_DIR = "./Generated_L128"
DEVICE = "cuda"
```

### Step 2: Run training

```bash
python main.py
```

---

## 📊 Outputs

### CSV Files
- metrics_global.csv  
- metrics_per_temp_bin.csv  
- preds_test.csv  
- training_times.csv  

### Model
- best_model.pth  

### PDF Reports
- physics_plots.pdf  
- training_plots.pdf  
- configs_bw.pdf  
- accuracy_vs_temperature.pdf  

---

## 📈 Evaluation Metrics

- Temperature MAE  
- Phase Accuracy  
- Reconstruction Accuracy:
  - Missing spins  
  - Observed spins  
  - Overall  

---

## 🧪 Applications

- Statistical physics inverse problems  
- Monte Carlo data correction  
- Scientific machine learning  
- Physics-informed AI  

---

## The Key Idea

Reconstruction must preserve **physical laws**, not just visual similarity.

---

##  Future Work

- Extension to 3D Ising model  
- Diffusion-based generative models  
- Super-resolution (L=32 → 64 → 128)  
- Transformer-based architectures  

---
##  Author

Dr. Partha Sarathi Bishnu  
Assistant Professor  
Department of CSE, BIT Mesra, India  

---

##  Support

If this project helps you, please  the repository and cite in your work.
