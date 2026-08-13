# -*- coding: utf-8 -*-
"""
Minimal Optimization and Physics-Loss Analysis
===================================================

This script is based on the supplied DLC2.py model/data pipeline and keeps the
EnhancedInvNet architecture unchanged.  It is specialized for the additional
experiments requested by Reviewer 3:

A. Reconstruction-only vs Full physics-guided comparison
B. Minimal convergence analysis
C. Three-point physics-loss-weight sensitivity

Default experiment:
    L = 32
    seeds = [123, 456, 789]

To rerun at another lattice size, either change L below or run, e.g.
    ISING_L=64 python ablation3.py

Expected input files:
    ../JOB5_Noise/J5Data/MCD32.csv
    ../JOB5_Noise/J5Data/MCDN32.csv

Important methodological detail
-------------------------------
C1 is retained as an EVALUATION observable, but is not used as an independent
training loss because, for the nearest-neighbor Ising model with J=1, it is
directly proportional to the energy. C2 at axial separation two lattice spacings
is used as the independent spatial-correlation regularizer.

The training physics observables are evaluated on the continuous tanh output
so that their gradients are well-defined.  Test/validation physical errors are
reported after binarizing the reconstructed spins to {-1,+1}.

Main outputs in Ablation3_L{L}/
-------------------------------
ablation_seed_results.csv
ablation_summary.csv
ablation_summary.tex
ablation_history.csv
sensitivity_seed_results.csv
sensitivity_summary.csv
sensitivity_summary.tex
sensitivity_history.csv
gradient_cosine_by_epoch.csv
gradient_cosine_summary.csv
gradient_cosine_summary.tex
convergence_comparison.pdf
gradient_cosine_heatmap.pdf
loss_weight_sensitivity.pdf
paper_ready_results.txt
run_manifest.txt
"""

from pathlib import Path
from collections import OrderedDict
import copy
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm


# =========================================================
# USER / HPC PARAMETERS
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# Default L=32.  For later runs:
#   ISING_L=64 python ablation3.py
L = int(os.environ.get("ISING_L", "128"))
Tc = 2.269

DATA_DIR = (SCRIPT_DIR / "../JOB5_Noise/J5Data").resolve()
CLEAN_CSV = DATA_DIR / f"MCD{L}.csv"
NOISY_CSV = DATA_DIR / f"MCDN{L}.csv"
OUT_DIR = SCRIPT_DIR / f"Ablation3_L{L}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 15
BATCH_SIZE = {32: 64, 64: 16, 128: 4}[L]
LR = 3e-4
PATIENCE = 3
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
DEVICE_PREF = "cuda"

# Fixed data split; model initialization / minibatch order varies across seeds.
SPLIT_SEED = 123
SEEDS = [123]
# Original/base loss weights.
LAM_REC = 2.0
LAM_BIN = 0.5
LAM_E = 0.1
LAM_M = 0.1
LAM_C2 = 0.1
LAM_T = 1.0
LAM_PHASE = 1.0

# Reviewer-requested sensitivity.
PHYS_LAMBDAS = [0.05, 0.10, 0.20]
# Gradient cosine is measured on one fixed training probe batch per epoch,
# over the SHARED ENCODER parameters.
GRAD_COSINE_EVERY = 1

# Full reviewer-compliant execution switches.
RUN_ABLATION = True
RUN_SENSITIVITY = True
RUN_GRADIENT_ANALYSIS = False
# For reproducibility. Set False only if maximum GPU speed is essential.
DETERMINISTIC = True


# =========================================================
# REPRODUCIBILITY / DEVICE
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def setup_device():
    if DEVICE_PREF == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return dev
    print("Using CPU")
    return torch.device("cpu")


device = setup_device()


# =========================================================
# DATA
# =========================================================
def _extract_spin_columns(df, L):
    spin_cols = [c for c in df.columns if c.lower().startswith("spin")]
    if not spin_cols:
        raise ValueError("No spin columns found (expected Spin* or spin_*).")

    def suffix_idx(name):
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 0

    spin_cols = sorted(spin_cols, key=suffix_idx)
    if len(spin_cols) != L * L:
        raise ValueError(
            f"Expected {L*L} spin columns for L={L}, found {len(spin_cols)}."
        )
    return spin_cols


def _phase_to_int(series):
    if np.issubdtype(series.dtype, np.number):
        return series.astype(np.int64)
    mapping = {"F": 1, "P": 0, "f": 1, "p": 0}
    mapped = series.map(mapping)
    if mapped.isna().any():
        raise ValueError(f"Unexpected phase labels: {series.unique()}")
    return mapped.astype(np.int64)


def load_data():
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"Clean CSV not found: {CLEAN_CSV}")
    if not NOISY_CSV.exists():
        raise FileNotFoundError(f"Noisy CSV not found: {NOISY_CSV}")

    print(f"Loading:\n  {CLEAN_CSV}\n  {NOISY_CSV}")
    clean = pd.read_csv(CLEAN_CSV)
    noisy = pd.read_csv(NOISY_CSV)

    spin_cols = _extract_spin_columns(clean, L)

    if len(clean) != len(noisy):
        raise ValueError(
            f"Clean/noisy row count mismatch: {len(clean)} vs {len(noisy)}"
        )

    T = clean["Temperature"].to_numpy(dtype=np.float32)
    P = _phase_to_int(clean["Phase"]).to_numpy(dtype=np.int64)

    S = clean[spin_cols].to_numpy(dtype=np.float32).reshape(-1, L, L)
    Y = noisy[spin_cols].to_numpy(dtype=np.float32).reshape(-1, L, L)
    M = (Y != 0).astype(np.float32)

    X = np.stack([Y, M], axis=1).astype(np.float32)

    return X, S.astype(np.float32), T, P, Y, M


class IsingDataset(Dataset):
    def __init__(self, X, S, T, P, Y, M):
        self.X = torch.from_numpy(X)
        self.S = torch.from_numpy(S)[:, None, ...]
        self.T = torch.from_numpy(T)[:, None]
        self.P = torch.from_numpy(P)
        self.Y = torch.from_numpy(Y)[:, None, ...]
        self.M = torch.from_numpy(M)[:, None, ...]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "S": self.S[idx],
            "T": self.T[idx],
            "P": self.P[idx],
            "Y": self.Y[idx],
            "M": self.M[idx],
        }


X_np, S_np, T_np, P_np, Y_np, M_np = load_data()
dataset = IsingDataset(X_np, S_np, T_np, P_np, Y_np, M_np)
N = len(dataset)

# One fixed 80/10/10 split for all seeds/configurations.
split_generator = torch.Generator().manual_seed(SPLIT_SEED)
perm = torch.randperm(N, generator=split_generator).tolist()
n_train = int(0.8 * N)
n_val = int(0.1 * N)

TRAIN_IDX = perm[:n_train]
VAL_IDX = perm[n_train:n_train + n_val]
TEST_IDX = perm[n_train + n_val:]

print(
    f"Dataset: N={N:,}, L={L}; "
    f"train={len(TRAIN_IDX):,}, val={len(VAL_IDX):,}, test={len(TEST_IDX):,}"
)


def make_loaders(seed):
    train_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, TRAIN_IDX),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_gen,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, VAL_IDX),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        Subset(dataset, TEST_IDX),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # Fixed probe batch used only for gradient-interaction diagnostics.
    probe_loader = DataLoader(
        Subset(dataset, TRAIN_IDX[:min(BATCH_SIZE, len(TRAIN_IDX))]),
        batch_size=min(BATCH_SIZE, len(TRAIN_IDX)),
        shuffle=False,
        num_workers=0,
    )
    probe_batch = next(iter(probe_loader))
    return train_loader, val_loader, test_loader, probe_batch


# =========================================================
# PHYSICS HELPERS
# =========================================================
def nn_sum(s):
    """Four nearest neighbors with periodic boundary conditions."""
    return (
        torch.roll(s, shifts=1, dims=2)
        + torch.roll(s, shifts=-1, dims=2)
        + torch.roll(s, shifts=1, dims=3)
        + torch.roll(s, shifts=-1, dims=3)
    )


def c2_sum(s):
    """Four axial second neighbors at separation two with periodic boundaries."""
    return (
        torch.roll(s, shifts=2, dims=2)
        + torch.roll(s, shifts=-2, dims=2)
        + torch.roll(s, shifts=2, dims=3)
        + torch.roll(s, shifts=-2, dims=3)
    )


# --- Differentiable observables used in TRAINING --------------------
def energy_soft(s):
    # J=1, each bond counted twice by nn_sum -> divide by 2.
    return -(s * nn_sum(s)).mean(dim=(1, 2, 3)) / 2.0


def mag_abs_soft(s):
    return s.mean(dim=(2, 3)).abs().squeeze(1)


def c2_soft(s):
    return (s * c2_sum(s)).mean(dim=(1, 2, 3)) / 4.0


# --- Discrete observables used in VALIDATION / TEST reporting -------
# C1 is evaluation-only; C2 is the independent correlation regularizer.
def binarize(s):
    return torch.where(s >= 0, torch.ones_like(s), -torch.ones_like(s))


def energy_discrete(s):
    s = binarize(s)
    return -(s * nn_sum(s)).mean(dim=(1, 2, 3)) / 2.0


def mag_abs_discrete(s):
    s = binarize(s)
    return s.mean(dim=(2, 3)).abs().squeeze(1)


def c1_discrete(s):
    s = binarize(s)
    return (s * nn_sum(s)).mean(dim=(1, 2, 3)) / 4.0


def c2_discrete(s):
    s = binarize(s)
    return (s * c2_sum(s)).mean(dim=(1, 2, 3)) / 4.0


def binary_consistency_loss(pred):
    return torch.mean((pred.abs() - 1.0).pow(2))


# =========================================================
# MODEL -- KEPT CONSISTENT WITH DLC2.py
# =========================================================
class EnhancedInvNet(nn.Module):
    def __init__(self, L):
        super().__init__()
        C = 64

        self.enc_conv1 = nn.Conv2d(2, C, 3, padding=1, padding_mode="circular")
        self.enc_conv2 = nn.Conv2d(C, C, 3, padding=1, padding_mode="circular")
        self.enc_conv3 = nn.Conv2d(C, C, 3, padding=1, padding_mode="circular")
        self.enc_conv4 = nn.Conv2d(C, C, 3, padding=1, padding_mode="circular")
        self.enc_conv5 = nn.Conv2d(C, C, 3, padding=1, padding_mode="circular")

        self.attention = nn.Sequential(
            nn.Conv2d(C, C // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 4, 1, 1),
            nn.Sigmoid(),
        )

        self.dec_conv1 = nn.Conv2d(C + 1, 64, 3, padding=1, padding_mode="circular")
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1, padding_mode="circular")
        self.dec_conv3 = nn.Conv2d(64, 32, 3, padding=1, padding_mode="circular")
        self.dec_conv4 = nn.Conv2d(32, 16, 3, padding=1, padding_mode="circular")
        self.dec_conv5 = nn.Conv2d(16, 1, 1)

        self.head_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.head_P = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        noisy_input = x[:, :1]
        mask = x[:, 1:]

        h1 = self.activation(self.enc_conv1(x))
        h2 = self.activation(self.enc_conv2(h1)) + h1
        h3 = self.activation(self.enc_conv3(h2)) + h2
        h4 = self.activation(self.enc_conv4(h3)) + h3
        h5 = self.activation(self.enc_conv5(h4)) + h4

        attention_map = self.attention(h5)
        enhanced_features = torch.cat([h5, attention_map], dim=1)

        d1 = self.activation(self.dec_conv1(enhanced_features))
        d2 = self.activation(self.dec_conv2(d1)) + d1
        d3 = self.activation(self.dec_conv3(d2))
        d4 = self.activation(self.dec_conv4(d3))
        S_hat = torch.tanh(self.dec_conv5(d4))

        # Preserve observed sites, as in the supplied model.
        S_final = mask * noisy_input + (1 - mask) * S_hat

        T_hat = self.head_T(h5)
        P_log = self.head_P(h5)
        return S_final, T_hat, P_log, attention_map


CE = nn.CrossEntropyLoss()


# =========================================================
# LOSS CONFIGURATIONS
# =========================================================
def W(rec=0.0, bin_=0.0, E=0.0, M=0.0, C2=0.0, T=0.0, Phi=0.0):
    return {
        "rec": float(rec),
        "bin": float(bin_),
        "E": float(E),
        "M": float(M),
        "C2": float(C2),
        "T": float(T),
        "Phi": float(Phi),
    }


# Term-isolation design: each intermediate model adds one reviewer-requested
# component to the same data-driven reconstruction objective.  "Full" contains
# the complete objective.  This isolates the contribution of each term more
# cleanly than a cumulative sequence.
ABLATION_CONFIGS = OrderedDict([
    ("RecOnly", W(rec=LAM_REC)),
    (
        "Full",
        W(
            rec=LAM_REC,
            bin_=LAM_BIN,
            E=LAM_E,
            M=LAM_M,
            C2=LAM_C2,
            T=LAM_T,
            Phi=LAM_PHASE,
        ),
    ),
])


def sensitivity_weights(lambda_phys):
    # Only the independent physical regularizers are swept.
    # Reconstruction, binary consistency, temperature and phase retain
    # their baseline weights.
    return W(
        rec=LAM_REC,
        bin_=LAM_BIN,
        E=lambda_phys,
        M=lambda_phys,
        C2=lambda_phys,
        T=LAM_T,
        Phi=LAM_PHASE,
    )


# =========================================================
# LOSS / METRIC FUNCTIONS
# =========================================================
def move_batch(batch):
    return {
        k: v.to(device, non_blocking=True)
        for k, v in batch.items()
    }


def compute_components(net, batch):
    Xb = batch["X"]
    S = batch["S"]
    Tt = batch["T"]
    P = batch["P"]
    Yb = batch["Y"]
    M = batch["M"]

    S_hat, T_hat, P_log, _ = net(Xb)

    observed_rec = (M * (S_hat - Yb).abs()).sum() / (M.sum() + 1e-8)
    missing_rec = ((1 - M) * (S_hat - S).abs()).sum() / (
        (1 - M).sum() + 1e-8
    )
    rec = observed_rec + 2.0 * missing_rec

    # Batch-level thermodynamic matching, consistent with the supplied code,
    # but evaluated with differentiable soft observables.
    L_E = (
        energy_soft(S_hat).mean() - energy_soft(S).mean().detach()
    ).pow(2)

    L_M = (
        mag_abs_soft(S_hat).mean() - mag_abs_soft(S).mean().detach()
    ).pow(2)

    L_C2 = (
        c2_soft(S_hat).mean() - c2_soft(S).mean().detach()
    ).pow(2)

    L_bin = binary_consistency_loss(S_hat)
    L_T = (T_hat - Tt).abs().mean()
    L_Phi = CE(P_log, P)

    return {
        "rec": rec,
        "bin": L_bin,
        "E": L_E,
        "M": L_M,
        "C2": L_C2,
        "T": L_T,
        "Phi": L_Phi,
    }, S_hat, T_hat, P_log


def weighted_total(components, weights):
    total = torch.zeros((), device=device)
    for key, value in components.items():
        total = total + weights[key] * value
    return total


def gradient_norm(model):
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            gnorm = p.grad.detach().norm(2).item()
            total_sq += gnorm * gnorm
    return math.sqrt(total_sq)


def batch_discrete_metrics(S_hat, S, M):
    S_pred = binarize(S_hat)
    S_true = binarize(S)

    missing = (M == 0)
    missing_den = max(1, int(missing.sum().item()))
    missing_correct = int(((S_pred == S_true) & missing).sum().item())

    overall_correct = int((S_pred == S_true).sum().item())
    overall_den = S_pred.numel()

    E_err = torch.abs(energy_discrete(S_pred) - energy_discrete(S_true))
    M_err = torch.abs(mag_abs_discrete(S_pred) - mag_abs_discrete(S_true))
    C1_err = torch.abs(c1_discrete(S_pred) - c1_discrete(S_true))
    C2_err = torch.abs(c2_discrete(S_pred) - c2_discrete(S_true))

    return {
        "missing_correct": missing_correct,
        "missing_den": missing_den,
        "overall_correct": overall_correct,
        "overall_den": overall_den,
        "E_abs_sum": float(E_err.sum().item()),
        "M_abs_sum": float(M_err.sum().item()),
        "C1_abs_sum": float(C1_err.sum().item()),
        "C2_abs_sum": float(C2_err.sum().item()),
        "n_samples": int(S.shape[0]),
    }


# =========================================================
# GRADIENT INTERACTION
# =========================================================
def shared_encoder_parameters(net):
    params = []
    for name, p in net.named_parameters():
        if name.startswith("enc_conv") and p.requires_grad:
            params.append(p)
    return params


def flattened_grad(loss, params, retain_graph=True):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )
    pieces = []
    for p, g in zip(params, grads):
        if g is None:
            pieces.append(torch.zeros_like(p).reshape(-1))
        else:
            pieces.append(g.detach().reshape(-1))
    return torch.cat(pieces)


def gradient_cosine_matrix(net, probe_batch):
    """
    Cosine similarities are computed for unweighted task losses over the
    shared encoder parameters.  Positive rho indicates locally aligned
    optimization directions; negative rho indicates conflict.
    """
    net.train()
    batch = move_batch(probe_batch)
    components, _, _, _ = compute_components(net, batch)

    loss_names = ["rec", "E", "M", "C2", "T", "Phi"]
    params = shared_encoder_parameters(net)

    grad_vecs = {}
    for i, name in enumerate(loss_names):
        grad_vecs[name] = flattened_grad(
            components[name],
            params,
            retain_graph=(i < len(loss_names) - 1),
        )

    rows = []
    for a in loss_names:
        ga = grad_vecs[a]
        na = torch.norm(ga).item()
        for b in loss_names:
            gb = grad_vecs[b]
            nb = torch.norm(gb).item()
            if na < 1e-12 or nb < 1e-12:
                rho = np.nan
            else:
                rho = float(
                    torch.dot(ga, gb).item() / (na * nb + 1e-12)
                )
            rows.append({"loss_a": a, "loss_b": b, "rho": rho})

    net.zero_grad(set_to_none=True)
    return rows


# =========================================================
# VALIDATION / TEST
# =========================================================
def validate_epoch(net, loader, weights):
    net.eval()

    comp_sum = {k: 0.0 for k in ["rec", "bin", "E", "M", "C2", "T", "Phi"]}
    total_loss_sum = 0.0
    n_batches = 0

    miss_correct = miss_den = 0
    overall_correct = overall_den = 0
    E_abs = M_abs = C1_abs = C2_abs = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch)
            components, S_hat, _, _ = compute_components(net, batch)
            total = weighted_total(components, weights)

            total_loss_sum += float(total.item())
            for k in comp_sum:
                comp_sum[k] += float(components[k].item())
            n_batches += 1

            dm = batch_discrete_metrics(S_hat, batch["S"], batch["M"])
            miss_correct += dm["missing_correct"]
            miss_den += dm["missing_den"]
            overall_correct += dm["overall_correct"]
            overall_den += dm["overall_den"]
            E_abs += dm["E_abs_sum"]
            M_abs += dm["M_abs_sum"]
            C1_abs += dm["C1_abs_sum"]
            C2_abs += dm["C2_abs_sum"]
            n_samples += dm["n_samples"]

    out = {
        "val_total": total_loss_sum / max(1, n_batches),
        "val_missing_acc": miss_correct / max(1, miss_den),
        "val_overall_acc": overall_correct / max(1, overall_den),
        "val_E_MAE": E_abs / max(1, n_samples),
        "val_M_MAE": M_abs / max(1, n_samples),
        "val_C1_MAE": C1_abs / max(1, n_samples),
        "val_C2_MAE": C2_abs / max(1, n_samples),
    }
    for k in comp_sum:
        out[f"val_{k}"] = comp_sum[k] / max(1, n_batches)
    return out


def evaluate_test(net, loader):
    net.eval()

    T_abs = 0.0
    phase_correct = 0
    n_samples = 0

    miss_correct = miss_den = 0
    overall_correct = overall_den = 0
    E_abs = M_abs = C1_abs = C2_abs = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch)
            S_hat, T_hat, P_log, _ = net(batch["X"])

            bs = int(batch["S"].shape[0])
            T_abs += float(torch.abs(T_hat - batch["T"]).sum().item())
            phase_correct += int(
                (P_log.argmax(dim=1) == batch["P"]).sum().item()
            )
            n_samples += bs

            dm = batch_discrete_metrics(S_hat, batch["S"], batch["M"])
            miss_correct += dm["missing_correct"]
            miss_den += dm["missing_den"]
            overall_correct += dm["overall_correct"]
            overall_den += dm["overall_den"]
            E_abs += dm["E_abs_sum"]
            M_abs += dm["M_abs_sum"]
            C1_abs += dm["C1_abs_sum"]
            C2_abs += dm["C2_abs_sum"]

    return {
        "Missing_Accuracy": miss_correct / max(1, miss_den),
        "Overall_Accuracy": overall_correct / max(1, overall_den),
        "Energy_MAE": E_abs / max(1, n_samples),
        "Magnetization_MAE": M_abs / max(1, n_samples),
        "C1_MAE": C1_abs / max(1, n_samples),
        "C2_MAE": C2_abs / max(1, n_samples),
        "Temperature_MAE": T_abs / max(1, n_samples),
        "Phase_Accuracy": phase_correct / max(1, n_samples),
        "N_test": n_samples,
    }


# =========================================================
# SINGLE TRAINING RUN
# =========================================================
def train_one(
    run_label,
    weights,
    seed,
    experiment_group,
    do_gradient_analysis=False,
):
    print("\n" + "=" * 88)
    print(
        f"{experiment_group.upper()} | {run_label} | seed={seed} | "
        f"L={L} | weights={weights}"
    )
    print("=" * 88)

    set_seed(seed)
    train_loader, val_loader, test_loader, probe_batch = make_loaders(seed)

    net = EnhancedInvNet(L).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=LR,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    patience_counter = 0

    history = []
    grad_rows = []

    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        net.train()

        train_total = 0.0
        train_comp = {k: 0.0 for k in ["rec", "bin", "E", "M", "C2", "T", "Phi"]}
        grad_norm_sum = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"{run_label} s{seed} e{epoch:02d}",
            leave=False,
            ncols=120,
        )

        for batch in pbar:
            batch = move_batch(batch)
            components, _, _, _ = compute_components(net, batch)
            total = weighted_total(components, weights)

            optimizer.zero_grad(set_to_none=True)
            total.backward()

            # Raw norm BEFORE clipping, as requested for optimization analysis.
            raw_gnorm = gradient_norm(net)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()

            train_total += float(total.item())
            for k in train_comp:
                train_comp[k] += float(components[k].item())
            grad_norm_sum += raw_gnorm
            n_batches += 1

            pbar.set_postfix(
                loss=f"{float(total.item()):.4f}",
                rec=f"{float(components['rec'].item()):.4f}",
                g=f"{raw_gnorm:.3f}",
            )

        train_total /= max(1, n_batches)
        for k in train_comp:
            train_comp[k] /= max(1, n_batches)
        mean_grad_norm = grad_norm_sum / max(1, n_batches)

        val = validate_epoch(net, val_loader, weights)

        row = {
            "Experiment": experiment_group,
            "Configuration": run_label,
            "Seed": seed,
            "Epoch": epoch,
            "Train_Total": train_total,
            "Train_Grad_Norm": mean_grad_norm,
            "Learning_Rate": scheduler.get_last_lr()[0],
        }
        for k, v in train_comp.items():
            row[f"Train_{k}"] = v
        row.update(val)
        history.append(row)

        # Gradient interactions are collected for the Full ablation model.
        if (
            do_gradient_analysis
            and RUN_GRADIENT_ANALYSIS
            and epoch % GRAD_COSINE_EVERY == 0
        ):
            gc = gradient_cosine_matrix(net, probe_batch)
            for r in gc:
                r.update(
                    {
                        "Experiment": experiment_group,
                        "Configuration": run_label,
                        "Seed": seed,
                        "Epoch": epoch,
                    }
                )
                grad_rows.append(r)

        if val["val_total"] < best_val:
            best_val = val["val_total"]
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in net.state_dict().items()
            }
        else:
            patience_counter += 1

        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_total:.5f} | "
            f"val={val['val_total']:.5f} | "
            f"miss_acc={val['val_missing_acc']:.4f} | "
            f"E={val['val_E_MAE']:.4f} | "
            f"M={val['val_M_MAE']:.4f} | "
            f"C2={val['val_C2_MAE']:.4f} | "
            f"grad={mean_grad_norm:.4f}"
        )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    runtime = time.time() - t0

    if best_state is not None:
        net.load_state_dict(best_state)

    test_metrics = evaluate_test(net, test_loader)

    final_grad = history[-1]["Train_Grad_Norm"] if history else np.nan

    result = {
        "Experiment": experiment_group,
        "Configuration": run_label,
        "Seed": seed,
        "L": L,
        **test_metrics,
        "Epoch_Min_Val_Loss": best_epoch,
        "Best_Val_Loss": best_val,
        "Final_Gradient_Norm": final_grad,
        "Epochs_Completed": len(history),
        "Runtime_Minutes": runtime / 60.0,
        "w_rec": weights["rec"],
        "w_bin": weights["bin"],
        "w_E": weights["E"],
        "w_M": weights["M"],
        "w_C2": weights["C2"],
        "w_T": weights["T"],
        "w_Phi": weights["Phi"],
    }

    del net, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result, history, grad_rows


# =========================================================
# SUMMARY HELPERS
# =========================================================
RESULT_METRICS = [
    "Missing_Accuracy",
    "Overall_Accuracy",
    "Energy_MAE",
    "Magnetization_MAE",
    "C2_MAE",
    "Temperature_MAE",
    "Phase_Accuracy",
    "Epoch_Min_Val_Loss",
    "Final_Gradient_Norm",
]


def summarize(df, group_col):
    rows = []
    for key, g in df.groupby(group_col, sort=False):
        row = {group_col: key, "n_seeds": int(len(g))}
        for metric in RESULT_METRICS:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = (
                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def pm(mean, std, decimals=3):
    if pd.isna(mean):
        return "--"
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def write_ablation_latex(summary_df, path):
    cols = [
        ("Missing_Accuracy", "Missing Acc."),
        ("Overall_Accuracy", "Overall Acc."),
        ("Energy_MAE", "$E$ MAE"),
        ("Magnetization_MAE", "$|M|$ MAE"),
        ("C2_MAE", "$C_2$ MAE"),
        ("Temperature_MAE", "$T$ MAE"),
        ("Phase_Accuracy", "Phase Acc."),
    ]

    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Minimal optimization comparison between reconstruction-only and full physics-guided learning for $L=%d$.}" % L,
        "\\label{tab:reviewer3_minimal}",
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\hline",
        "Configuration & " + " & ".join(label for _, label in cols) + " \\\\",
        "\\hline",
    ]

    for _, r in summary_df.iterrows():
        values = [
            pm(r[f"{metric}_mean"], r[f"{metric}_std"], 3)
            for metric, _ in cols
        ]
        lines.append(
            str(r["Configuration"]) + " & " + " & ".join(values) + " \\\\"
        )

    lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_sensitivity_latex(summary_df, path):
    cols = [
        ("Missing_Accuracy", "Missing Acc."),
        ("Energy_MAE", "$E$ MAE"),
        ("Magnetization_MAE", "$|M|$ MAE"),
        ("C2_MAE", "$C_2$ MAE"),
        ("Temperature_MAE", "$T$ MAE"),
        ("Phase_Accuracy", "Phase Acc."),
    ]

    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Minimal sensitivity analysis for the common physics-loss coefficient at $L=%d$.}" % L,
        "\\label{tab:reviewer3_minimal_sensitivity}",
        "\\begin{tabular}{c" + "c" * len(cols) + "}",
        "\\hline",
        "$\\lambda_{\\mathrm{phys}}$ & " + " & ".join(label for _, label in cols) + " \\\\",
        "\\hline",
    ]

    for _, r in summary_df.iterrows():
        values = [
            pm(r[f"{metric}_mean"], r[f"{metric}_std"], 3)
            for metric, _ in cols
        ]
        lines.append(
            f"{float(r['Lambda_Phys']):.2f} & " + " & ".join(values) + " \\\\"
        )

    lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# PLOTTING
# =========================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})


def aggregate_history(history_df, config, column):
    d = history_df[history_df["Configuration"] == config]
    grp = d.groupby("Epoch")[column]
    out = pd.DataFrame({
        "Epoch": grp.mean().index,
        "mean": grp.mean().values,
        "std": grp.std(ddof=1).fillna(0.0).values,
    })
    return out


def add_curve(ax, agg, label):
    ax.plot(agg["Epoch"], agg["mean"], label=label)
    ax.fill_between(
        agg["Epoch"],
        agg["mean"] - agg["std"],
        agg["mean"] + agg["std"],
        alpha=0.15,
    )


def make_convergence_pdf(history_df, out_pdf):
    """
    Minimal reviewer-facing convergence figure:
    1) validation total loss
    2) validation missing-spin accuracy
    3) validation E, |M| and C2 errors
    """
    with PdfPages(out_pdf) as pdf:
        # Page 1: validation loss
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for cfg in ["RecOnly", "Full"]:
            agg = aggregate_history(history_df, cfg, "val_total")
            if len(agg):
                add_curve(ax, agg, cfg)
        ax.set_title("Validation convergence")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: missing-spin accuracy
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for cfg in ["RecOnly", "Full"]:
            agg = aggregate_history(history_df, cfg, "val_missing_acc")
            if len(agg):
                add_curve(ax, agg, cfg)
        ax.set_title("Missing-spin reconstruction accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Full-model physical errors
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for col, label in [
            ("val_E_MAE", "$E$ MAE"),
            ("val_M_MAE", "$|M|$ MAE"),
            ("val_C2_MAE", "$C_2$ MAE"),
        ]:
            agg = aggregate_history(history_df, "Full", col)
            if len(agg):
                ax.plot(agg["Epoch"], agg["mean"], label=label)
        ax.set_title("Physical-consistency convergence (Full model)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def make_gradient_heatmap(grad_df, out_pdf):
    if grad_df.empty:
        return

    names = ["rec", "E", "M", "C2", "T", "Phi"]
    mean_df = (
        grad_df.groupby(["loss_a", "loss_b"], as_index=False)["rho"]
        .mean()
    )
    matrix = np.full((len(names), len(names)), np.nan, dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            m = mean_df[
                (mean_df["loss_a"] == a)
                & (mean_df["loss_b"] == b)
            ]
            if len(m):
                matrix[i, j] = float(m["rho"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_title("Average gradient cosine similarity\n(shared encoder parameters)")

    for i in range(len(names)):
        for j in range(len(names)):
            if not np.isnan(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    fig.colorbar(im, ax=ax, label="$\\rho_{a,b}$")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def make_sensitivity_pdf(summary_df, out_pdf):
    x = summary_df["Lambda_Phys"].to_numpy(dtype=float)

    pages = [
        ("Missing_Accuracy", "Missing-spin accuracy", "Accuracy"),
        ("Overall_Accuracy", "Overall reconstruction accuracy", "Accuracy"),
        ("Energy_MAE", "Energy error", "$E$ MAE"),
        ("Magnetization_MAE", "Magnetization error", "$|M|$ MAE"),
        ("C2_MAE", "$C_2$ error", "$C_2$ MAE"),
        ("Temperature_MAE", "Temperature error", "$T$ MAE"),
        ("Phase_Accuracy", "Phase accuracy", "Accuracy"),
    ]

    with PdfPages(out_pdf) as pdf:
        for metric, title, ylabel in pages:
            y = summary_df[f"{metric}_mean"].to_numpy(dtype=float)
            e = summary_df[f"{metric}_std"].to_numpy(dtype=float)

            fig, ax = plt.subplots(figsize=(8, 5.5))
            ax.errorbar(x, y, yerr=e, marker="o", capsize=4)
            ax.set_xscale("symlog", linthresh=0.01)
            ax.set_xlabel("$\\lambda_{\\mathrm{phys}}$")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# =========================================================
# PAPER-READY TEXT
# =========================================================
def write_paper_ready_text(ablation_summary, sensitivity_summary, grad_df, path):
    lines = []
    lines.append(f"Reviewer 3 numerical summary for L={L}")
    lines.append("=" * 72)

    def get_cfg(name):
        return ablation_summary[
            ablation_summary["Configuration"] == name
        ].iloc[0]

    if (
        "RecOnly" in set(ablation_summary["Configuration"])
        and "Full" in set(ablation_summary["Configuration"])
    ):
        rec = get_cfg("RecOnly")
        full = get_cfg("Full")

        lines.append("")
        lines.append("Data-only versus full physics-guided model:")
        for metric in [
            "Missing_Accuracy",
            "Overall_Accuracy",
            "Energy_MAE",
            "Magnetization_MAE",
            "C2_MAE",
            "Temperature_MAE",
            "Phase_Accuracy",
            "Epoch_Min_Val_Loss",
            "Final_Gradient_Norm",
        ]:
            r = rec[f"{metric}_mean"]
            f = full[f"{metric}_mean"]
            lines.append(
                f"  {metric}: RecOnly={r:.6f}, Full={f:.6f}, "
                f"Full-RecOnly={f-r:+.6f}"
            )

    if not grad_df.empty:
        g = (
            grad_df[
                (grad_df["loss_a"] == "rec")
                & (grad_df["loss_b"].isin(["E", "M", "C2", "T", "Phi"]))
            ]
            .groupby("loss_b")["rho"]
            .agg(["mean", "std"])
        )
        lines.append("")
        lines.append("Mean gradient cosine similarity with reconstruction loss:")
        for name, row in g.iterrows():
            lines.append(
                f"  rho(rec,{name}) = {row['mean']:.6f} +/- {row['std']:.6f}"
            )

    if not sensitivity_summary.empty:
        lines.append("")
        lines.append("Physics-loss sensitivity:")
        for _, r in sensitivity_summary.iterrows():
            lines.append(
                f"  lambda_phys={r['Lambda_Phys']:.2f}: "
                f"MissingAcc={r['Missing_Accuracy_mean']:.6f}, "
                f"E_MAE={r['Energy_MAE_mean']:.6f}, "
                f"M_MAE={r['Magnetization_MAE_mean']:.6f}, "
                f"C2_MAE={r['C2_MAE_mean']:.6f}, "
                f"T_MAE={r['Temperature_MAE_mean']:.6f}, "
                f"PhaseAcc={r['Phase_Accuracy_mean']:.6f}"
            )

    lines.append("")
    lines.append(
        "Use the CSV/LaTeX tables for manuscript values. "
        "Do not replace these measured values with estimates."
    )

    Path(path).write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================
def main():
    start = time.time()
    print("\nReviewer 3 Optimization and Physics-Loss Analysis")
    print(f"L={L}; seeds={SEEDS}; epochs={EPOCHS}; batch={BATCH_SIZE}")
    print(f"Output: {OUT_DIR}")

    ablation_results = []
    ablation_history = []
    gradient_rows = []

    full_result_by_seed = {}
    full_history_by_seed = {}

    # -----------------------------------------------------
    # A. ABLATION
    # -----------------------------------------------------
    if RUN_ABLATION:
        for cfg, weights in ABLATION_CONFIGS.items():
            for seed in SEEDS:
                result, history, grad = train_one(
                    run_label=cfg,
                    weights=weights,
                    seed=seed,
                    experiment_group="ablation",
                    do_gradient_analysis=(cfg == "Full"),
                )
                ablation_results.append(result)
                ablation_history.extend(history)
                gradient_rows.extend(grad)

                if cfg == "Full":
                    full_result_by_seed[seed] = copy.deepcopy(result)
                    full_history_by_seed[seed] = copy.deepcopy(history)

        ablation_df = pd.DataFrame(ablation_results)
        ablation_hist_df = pd.DataFrame(ablation_history)
        grad_df = pd.DataFrame(gradient_rows)

        ablation_df.to_csv(
            OUT_DIR / "ablation_seed_results.csv",
            index=False,
        )
        ablation_hist_df.to_csv(
            OUT_DIR / "ablation_history.csv",
            index=False,
        )

        ablation_summary = summarize(ablation_df, "Configuration")
        ablation_summary.to_csv(
            OUT_DIR / "ablation_summary.csv",
            index=False,
        )
        write_ablation_latex(
            ablation_summary,
            OUT_DIR / "ablation_summary.tex",
        )

        make_convergence_pdf(
            ablation_hist_df,
            OUT_DIR / "convergence_comparison.pdf",
        )

        if not grad_df.empty:
            grad_df.to_csv(
                OUT_DIR / "gradient_cosine_by_epoch.csv",
                index=False,
            )

            grad_summary = (
                grad_df.groupby(["loss_a", "loss_b"])["rho"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            grad_summary.to_csv(
                OUT_DIR / "gradient_cosine_summary.csv",
                index=False,
            )

            # Compact reviewer-requested reconstruction-vs-other table.
            rec_grad = grad_summary[
                (grad_summary["loss_a"] == "rec")
                & (grad_summary["loss_b"].isin(["E", "M", "C2", "T", "Phi"]))
            ].copy()

            tex = [
                "\\begin{table}[!t]",
                "\\centering",
                "\\caption{Average gradient cosine similarity between the reconstruction objective and auxiliary objectives for the Full model.}",
                "\\label{tab:gradient_cosine}",
                "\\begin{tabular}{lc}",
                "\\hline",
                "Loss pair & $\\rho$ \\\\",
                "\\hline",
            ]
            for _, r in rec_grad.iterrows():
                tex.append(
                    f"$\\rho_{{\\mathrm{{rec}},{r['loss_b']}}}$ & "
                    f"{r['mean']:.3f} $\\pm$ {r['std']:.3f} \\\\"
                )
            tex += ["\\hline", "\\end{tabular}", "\\end{table}"]
            (OUT_DIR / "gradient_cosine_summary.tex").write_text(
                "\n".join(tex),
                encoding="utf-8",
            )

            make_gradient_heatmap(
                grad_df,
                OUT_DIR / "gradient_cosine_heatmap.pdf",
            )
        else:
            grad_summary = pd.DataFrame()
            grad_df = pd.DataFrame()
    else:
        # Allows plotting/sensitivity reruns if prior CSVs already exist.
        ablation_df = pd.read_csv(OUT_DIR / "ablation_seed_results.csv")
        ablation_hist_df = pd.read_csv(OUT_DIR / "ablation_history.csv")
        ablation_summary = pd.read_csv(OUT_DIR / "ablation_summary.csv")
        grad_path = OUT_DIR / "gradient_cosine_by_epoch.csv"
        grad_df = pd.read_csv(grad_path) if grad_path.exists() else pd.DataFrame()

        for seed in SEEDS:
            m = (
                (ablation_df["Configuration"] == "Full")
                & (ablation_df["Seed"] == seed)
            )
            if m.any():
                full_result_by_seed[seed] = ablation_df[m].iloc[0].to_dict()

    # -----------------------------------------------------
    # D. LOSS-WEIGHT SENSITIVITY
    # -----------------------------------------------------
    sensitivity_results = []
    sensitivity_history = []

    if RUN_SENSITIVITY:
        for lam in PHYS_LAMBDAS:
            for seed in SEEDS:
                # lambda=0.1 is identical to Full -> reuse measured Full run.
                if abs(lam - LAM_E) < 1e-12 and seed in full_result_by_seed:
                    reused = copy.deepcopy(full_result_by_seed[seed])
                    reused["Experiment"] = "sensitivity"
                    reused["Configuration"] = f"lambda={lam:g}"
                    reused["Lambda_Phys"] = lam
                    sensitivity_results.append(reused)

                    if seed in full_history_by_seed:
                        for h in full_history_by_seed[seed]:
                            hh = copy.deepcopy(h)
                            hh["Experiment"] = "sensitivity"
                            hh["Configuration"] = f"lambda={lam:g}"
                            hh["Lambda_Phys"] = lam
                            sensitivity_history.append(hh)
                    continue

                weights = sensitivity_weights(lam)
                result, history, _ = train_one(
                    run_label=f"lambda={lam:g}",
                    weights=weights,
                    seed=seed,
                    experiment_group="sensitivity",
                    do_gradient_analysis=False,
                )
                result["Lambda_Phys"] = lam
                sensitivity_results.append(result)
                for h in history:
                    h["Lambda_Phys"] = lam
                    sensitivity_history.append(h)

        sensitivity_df = pd.DataFrame(sensitivity_results)
        sensitivity_hist_df = pd.DataFrame(sensitivity_history)

        sensitivity_df.to_csv(
            OUT_DIR / "sensitivity_seed_results.csv",
            index=False,
        )
        sensitivity_hist_df.to_csv(
            OUT_DIR / "sensitivity_history.csv",
            index=False,
        )

        rows = []
        for lam, g in sensitivity_df.groupby("Lambda_Phys", sort=True):
            row = {"Lambda_Phys": float(lam), "n_seeds": int(len(g))}
            for metric in RESULT_METRICS:
                vals = pd.to_numeric(g[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_std"] = (
                    float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                )
            rows.append(row)
        sensitivity_summary = pd.DataFrame(rows)

        sensitivity_summary.to_csv(
            OUT_DIR / "sensitivity_summary.csv",
            index=False,
        )
        write_sensitivity_latex(
            sensitivity_summary,
            OUT_DIR / "sensitivity_summary.tex",
        )
        make_sensitivity_pdf(
            sensitivity_summary,
            OUT_DIR / "loss_weight_sensitivity.pdf",
        )
    else:
        path = OUT_DIR / "sensitivity_summary.csv"
        sensitivity_summary = (
            pd.read_csv(path) if path.exists() else pd.DataFrame()
        )

    # -----------------------------------------------------
    # PAPER-READY NUMERICAL TEXT + MANIFEST
    # -----------------------------------------------------
    write_paper_ready_text(
        ablation_summary,
        sensitivity_summary,
        grad_df,
        OUT_DIR / "paper_ready_results.txt",
    )

    elapsed = time.time() - start

    manifest = f"""Reviewer 3 experiment manifest
Timestamp: {datetime.now().isoformat()}
L: {L}
Tc: {Tc}
Clean CSV: {CLEAN_CSV}
Noisy CSV: {NOISY_CSV}
N: {N}
Train/Val/Test: {len(TRAIN_IDX)}/{len(VAL_IDX)}/{len(TEST_IDX)}
Seeds: {SEEDS}
Split seed: {SPLIT_SEED}
Epochs: {EPOCHS}
Batch size: {BATCH_SIZE}
Learning rate: {LR}
Patience: {PATIENCE}
Base weights:
  rec={LAM_REC}
  bin={LAM_BIN}
  E={LAM_E}
  M={LAM_M}
  C2={LAM_C2}
  T={LAM_T}
  Phi={LAM_PHASE}
Physics sensitivity: {PHYS_LAMBDAS}
Gradient cosine: shared encoder parameters; every {GRAD_COSINE_EVERY} epoch(s)
C1: evaluation only (not used as an independent loss)
C2: training/sensitivity correlation regularizer; axial separation two with periodic boundary conditions
Runtime minutes: {elapsed/60.0:.3f}
"""
    (OUT_DIR / "run_manifest.txt").write_text(manifest, encoding="utf-8")

    print("\n" + "=" * 88)
    print("REVIEWER 3 ANALYSIS COMPLETED")
    print("=" * 88)
    print(f"Total runtime: {elapsed/60.0:.2f} min")
    print("Paper-ready outputs:")
    for name in [
        "ablation_summary.csv",
        "ablation_summary.tex",
        "convergence_comparison.pdf",
        "sensitivity_summary.csv",
        "sensitivity_summary.tex",
        "loss_weight_sensitivity.pdf",
        "paper_ready_results.txt",
        "run_manifest.txt",
    ]:
        p = OUT_DIR / name
        print(f"  {'OK' if p.exists() else '--'} {p}")


if __name__ == "__main__":
    main()
