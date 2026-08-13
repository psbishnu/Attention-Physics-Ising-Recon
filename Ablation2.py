# -*- coding: utf-8 -*-
"""
Ablation study runner for inverse inference on Ising (clean MCD{L}.csv + noisy MCDN{L}.csv)

Ablations included (ONLY requested ones):
1) Attention Mechanism Removal
2) Simpler Encoder/Decoder (3 conv blocks)
3) Vanilla CNN Baseline (no residual, no attention)
4) Single Physics Loss (E only / M only / C2 only)
5) Weight Tuning Ablation (remove weighting scheme -> all weights=1)
6) Mask Channel Removal (input only noisy spins)
7) Noise Level Ablation (generate noise masks at different missing percentages)
8) Channel Width Reduction (64 -> 32)
9) Channel Width Reduction (64 -> 16)

Outputs:
- ablation_L{L}/ablation_results.csv
- ablation_L{L}/ablation_summary.pdf  (B/W/gray, bold font size 16)
- ablation_L{L}/variant_<name>/metrics_global.csv, training_times.csv, best_model.pth
"""

# ==========================
# USER PARAMETERS
# ==========================
L = 32
Tc = 2.269

CLEAN_CSV = "./J5Data/MCD32.csv"   # path of the clean dataset
NOISY_CSV = "./J5Data/MCDN32.csv"  # path of the noisy dataset

ABLATION_DIR = f"./ablation_L{L}"

EPOCHS = 40
BATCH_SIZE = 64
LR = 3e-4
DEVICE = "cuda"
PATIENCE = 7
SEED = 123

# base weights (used in DEFAULT variant)
BASE_LAM_REC   = 2.0
BASE_LAM_E     = 0.1
BASE_LAM_M     = 0.1
BASE_LAM_C2    = 0.1
BASE_LAM_T     = 1.0
BASE_LAM_PHASE = 1.0
BASE_LAM_BIN   = 0.5

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

# Noise ablation missing rates (fraction of spins set to 0)
NOISE_LEVELS = [0.1, 0.3, 0.5, 0.7]   # 10%, 30%, 50%, 70% missing

DEBUG_MAX_SAMPLES = None  # set small for debug


# ==========================
# IMPORTS
# ==========================
import os, time, json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


# ==========================
# REPRODUCIBILITY
# ==========================
def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(SEED)


# ==========================
# DEVICE
# ==========================
def setup_device(device_preference="cuda"):
    if device_preference == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"? Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("?? Using CPU")
    return device


# ==========================
# DATA HELPERS
# ==========================
def _extract_spin_columns(df, L):
    spin_cands = [c for c in df.columns if c.lower().startswith("spin")]
    if not spin_cands:
        raise ValueError("No spin columns found (expected 'Spin1..' or 'spin_0..').")

    def _suffix_idx(name):
        s = "".join(ch for ch in name if ch.isdigit())
        return int(s) if s != "" else 0

    spin_sorted = sorted(spin_cands, key=_suffix_idx)
    if len(spin_sorted) != L * L:
        raise ValueError(f"Expected {L*L} spin cols, got {len(spin_sorted)}. Check L/CSV.")
    return spin_sorted


def _phase_to_int(series):
    if np.issubdtype(series.dtype, np.number):
        return series.astype(np.int64)
    mapping = {'F': 1, 'P': 0, 'f': 1, 'p': 0}
    mapped = series.map(mapping)
    if mapped.isna().any():
        raise ValueError(f"Unexpected Phase values: {series.unique()}")
    return mapped.astype(np.int64)


def load_clean_noisy(clean_path, noisy_path, L, max_samples=None):
    df_clean = pd.read_csv(clean_path)
    df_noisy = pd.read_csv(noisy_path)

    if max_samples is not None:
        df_clean = df_clean.iloc[:max_samples].reset_index(drop=True)
        df_noisy = df_noisy.iloc[:max_samples].reset_index(drop=True)

    spin_cols = _extract_spin_columns(df_clean, L)

    T  = df_clean["Temperature"].to_numpy(dtype=np.float32)
    Ph = _phase_to_int(df_clean["Phase"]).to_numpy(dtype=np.int64)

    Sigma = df_clean[spin_cols].to_numpy(dtype=np.int8).reshape(-1, L, L).astype(np.float32)
    Y0    = df_noisy[spin_cols].to_numpy(dtype=np.int8).reshape(-1, L, L).astype(np.float32)

    return Sigma, Y0, T, Ph


def make_noisy_from_clean(Sigma, missing_rate, seed=0):
    """
    Generate noise by masking a random fraction (missing_rate) of spins -> 0.
    Keeps observed spins unchanged.
    """
    rng = np.random.default_rng(seed)
    N, L1, L2 = Sigma.shape
    assert L1 == L2
    total = L1 * L2
    k = int(round(missing_rate * total))

    Y = Sigma.copy()
    Mask = np.ones_like(Sigma, dtype=np.float32)

    for i in range(N):
        idx = rng.choice(total, size=k, replace=False)
        r = idx // L1
        c = idx % L1
        Y[i, r, c] = 0.0
        Mask[i, r, c] = 0.0

    return Y, Mask


def build_input(Y, Mask, use_mask_channel=True):
    """
    Returns X:
      if use_mask_channel: X=[Y, Mask] -> (N,2,L,L)
      else: X=[Y] -> (N,1,L,L)
    """
    if use_mask_channel:
        X = np.stack([Y, Mask], axis=1).astype(np.float32)
    else:
        X = Y[:, None, ...].astype(np.float32)
    return X


class IsingDS(Dataset):
    def __init__(self, X, Sigma, T, Phase, Y, Mask):
        self.X = torch.from_numpy(X)                    # (N,Cin,L,L)
        self.S = torch.from_numpy(Sigma)[:, None, ...]  # (N,1,L,L)
        self.T = torch.from_numpy(T)[:, None]           # (N,1)
        self.P = torch.from_numpy(Phase)                # (N,)
        self.Y = torch.from_numpy(Y)[:, None, ...]      # (N,1,L,L)
        self.M = torch.from_numpy(Mask)[:, None, ...]   # (N,1,L,L)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return {"X": self.X[i], "S": self.S[i], "T": self.T[i], "P": self.P[i],
                "Y": self.Y[i], "M": self.M[i]}


# ==========================
# PHYSICS HELPERS
# ==========================
class PeriodicConv(nn.Module):
    def __init__(self, kernel_3x3):
        super().__init__()
        k = torch.tensor(kernel_3x3, dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("weight", k)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="circular")
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=0)


def make_physics_helpers(device):
    _NN_K = [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]]
    nn_conv = PeriodicConv(_NN_K).to(device)

    def energy_per_site(s):
        nbr = nn_conv(s)
        s_clamped = torch.sign(s)
        return (-(s_clamped * nbr)).mean(dim=(1, 2, 3)) / 2.0

    def nn_corr(s):
        # C1: nearest-neighbor correlation. Retained only as a helper/evaluation
        # observable; it is not used as an independent training loss because
        # e(S) = -2 C1(S) for the nearest-neighbor Ising model with J=1.
        nbr = nn_conv(s)
        s_clamped = torch.sign(s)
        return (s_clamped * nbr).mean(dim=(1, 2, 3)) / 4.0

    def c2_corr(s):
        # C2: second-neighbor correlation at axial separation two,
        # using periodic boundary conditions:
        # (u±2,v) and (u,v±2).
        s_clamped = torch.sign(s)
        nbr2 = (
            torch.roll(s, shifts= 2, dims=2)
            + torch.roll(s, shifts=-2, dims=2)
            + torch.roll(s, shifts= 2, dims=3)
            + torch.roll(s, shifts=-2, dims=3)
        )
        return (s_clamped * nbr2).mean(dim=(1, 2, 3)) / 4.0

    def mag_abs(s):
        s_clamped = torch.sign(s)
        return s_clamped.mean(dim=(2, 3)).abs().squeeze(1)

    return energy_per_site, nn_corr, c2_corr, mag_abs


def binary_consistency_loss(pred):
    return torch.mean((pred.abs() - 1.0).pow(2))


# ==========================
# MODEL VARIANTS
# ==========================
class ResidualBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.c1 = nn.Conv2d(C, C, 3, padding=1, padding_mode="circular")
        self.c2 = nn.Conv2d(C, C, 3, padding=1, padding_mode="circular")
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.c1(x))
        y = self.c2(y)
        return self.act(y + x)


class InvNet_Full(nn.Module):
    """Full: residual encoder (5 blocks), attention, decoder."""
    def __init__(self, Cin, C=64):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

        self.inp = nn.Conv2d(Cin, C, 3, padding=1, padding_mode="circular")
        self.rb1 = ResidualBlock(C)
        self.rb2 = ResidualBlock(C)
        self.rb3 = ResidualBlock(C)
        self.rb4 = ResidualBlock(C)

        self.att = nn.Sequential(
            nn.Conv2d(C, C//4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C//4, 1, 1), nn.Sigmoid()
        )

        self.dec1 = nn.Conv2d(C + 1, 64, 3, padding=1, padding_mode="circular")
        self.dec2 = nn.Conv2d(64, 64, 3, padding=1, padding_mode="circular")
        self.dec3 = nn.Conv2d(64, 32, 3, padding=1, padding_mode="circular")
        self.dec4 = nn.Conv2d(32, 16, 3, padding=1, padding_mode="circular")
        self.out  = nn.Conv2d(16, 1, 1)

        self.head_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.head_P = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        noisy_input = x[:, :1, :, :]  # first channel always Y
        mask = x[:, 1:, :, :] if x.shape[1] > 1 else None

        h = self.act(self.inp(x))
        h = self.rb1(h)
        h = self.rb2(h)
        h = self.rb3(h)
        h = self.rb4(h)

        att_map = self.att(h)
        hcat = torch.cat([h, att_map], dim=1)

        d = self.act(self.dec1(hcat))
        d = self.act(self.dec2(d))
        d = self.act(self.dec3(d))
        d = self.act(self.dec4(d))
        S_hat = torch.tanh(self.out(d))

        if mask is None:
            # no mask channel -> output direct prediction
            S_final = S_hat
        else:
            S_final = mask * noisy_input + (1 - mask) * S_hat

        T_hat = self.head_T(h)
        P_log = self.head_P(h)
        return S_final, T_hat, P_log, att_map


class InvNet_NoAttention(nn.Module):
    """1) Attention removed: no att module, decoder takes only h."""
    def __init__(self, Cin, C=64):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

        self.inp = nn.Conv2d(Cin, C, 3, padding=1, padding_mode="circular")
        self.rb1 = ResidualBlock(C)
        self.rb2 = ResidualBlock(C)
        self.rb3 = ResidualBlock(C)
        self.rb4 = ResidualBlock(C)

        self.dec1 = nn.Conv2d(C, 64, 3, padding=1, padding_mode="circular")
        self.dec2 = nn.Conv2d(64, 64, 3, padding=1, padding_mode="circular")
        self.dec3 = nn.Conv2d(64, 32, 3, padding=1, padding_mode="circular")
        self.dec4 = nn.Conv2d(32, 16, 3, padding=1, padding_mode="circular")
        self.out  = nn.Conv2d(16, 1, 1)

        self.head_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.head_P = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        noisy_input = x[:, :1, :, :]
        mask = x[:, 1:, :, :] if x.shape[1] > 1 else None

        h = self.act(self.inp(x))
        h = self.rb1(h)
        h = self.rb2(h)
        h = self.rb3(h)
        h = self.rb4(h)

        d = self.act(self.dec1(h))
        d = self.act(self.dec2(d))
        d = self.act(self.dec3(d))
        d = self.act(self.dec4(d))
        S_hat = torch.tanh(self.out(d))

        if mask is None:
            S_final = S_hat
        else:
            S_final = mask * noisy_input + (1 - mask) * S_hat

        T_hat = self.head_T(h)
        P_log = self.head_P(h)
        # attention map absent -> return zeros (for interface)
        return S_final, T_hat, P_log, torch.zeros_like(S_hat)


class InvNet_Simple3(nn.Module):
    """2) Simpler encoder/decoder: 3 conv blocks, no residual, optional attention."""
    def __init__(self, Cin, C=64, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.act = nn.ReLU(inplace=True)

        self.c1 = nn.Conv2d(Cin, C, 3, padding=1, padding_mode="circular")
        self.c2 = nn.Conv2d(C,  C, 3, padding=1, padding_mode="circular")
        self.c3 = nn.Conv2d(C,  C, 3, padding=1, padding_mode="circular")

        if use_attention:
            self.att = nn.Sequential(
                nn.Conv2d(C, C//4, 1), nn.ReLU(inplace=True),
                nn.Conv2d(C//4, 1, 1), nn.Sigmoid()
            )
            dec_in = C + 1
        else:
            self.att = None
            dec_in = C

        self.d1 = nn.Conv2d(dec_in, 32, 3, padding=1, padding_mode="circular")
        self.d2 = nn.Conv2d(32, 16, 3, padding=1, padding_mode="circular")
        self.out = nn.Conv2d(16, 1, 1)

        self.head_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self.head_P = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        noisy_input = x[:, :1, :, :]
        mask = x[:, 1:, :, :] if x.shape[1] > 1 else None

        h = self.act(self.c1(x))
        h = self.act(self.c2(h))
        h = self.act(self.c3(h))

        if self.att is not None:
            att_map = self.att(h)
            h = torch.cat([h, att_map], dim=1)
        else:
            att_map = torch.zeros_like(noisy_input)

        d = self.act(self.d1(h))
        d = self.act(self.d2(d))
        S_hat = torch.tanh(self.out(d))

        if mask is None:
            S_final = S_hat
        else:
            S_final = mask * noisy_input + (1 - mask) * S_hat

        T_hat = self.head_T(h[:, :64, :, :] if h.shape[1] > 64 else h)
        P_log = self.head_P(h[:, :64, :, :] if h.shape[1] > 64 else h)
        return S_final, T_hat, P_log, att_map


class InvNet_VanillaCNN(nn.Module):
    """3) Vanilla CNN baseline: no residual, no attention."""
    def __init__(self, Cin, C=64):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.c1 = nn.Conv2d(Cin, C, 3, padding=1, padding_mode="circular")
        self.c2 = nn.Conv2d(C,  C, 3, padding=1, padding_mode="circular")
        self.c3 = nn.Conv2d(C,  32, 3, padding=1, padding_mode="circular")
        self.c4 = nn.Conv2d(32, 16, 3, padding=1, padding_mode="circular")
        self.out = nn.Conv2d(16, 1, 1)

        self.head_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self.head_P = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        noisy_input = x[:, :1, :, :]
        mask = x[:, 1:, :, :] if x.shape[1] > 1 else None

        h = self.act(self.c1(x))
        h2 = self.act(self.c2(h))
        d = self.act(self.c3(h2))
        d = self.act(self.c4(d))
        S_hat = torch.tanh(self.out(d))

        if mask is None:
            S_final = S_hat
        else:
            S_final = mask * noisy_input + (1 - mask) * S_hat

        T_hat = self.head_T(h)
        P_log = self.head_P(h)
        return S_final, T_hat, P_log, torch.zeros_like(S_hat)


def build_model(model_key, Cin, C):
    if model_key == "FULL":
        return InvNet_Full(Cin=Cin, C=C)
    if model_key == "NO_ATT":
        return InvNet_NoAttention(Cin=Cin, C=C)
    if model_key == "SIMPLE3":
        return InvNet_Simple3(Cin=Cin, C=C, use_attention=True)
    if model_key == "SIMPLE3_NOATT":
        return InvNet_Simple3(Cin=Cin, C=C, use_attention=False)
    if model_key == "VANILLA":
        return InvNet_VanillaCNN(Cin=Cin, C=C)
    raise ValueError(f"Unknown model_key: {model_key}")


# ==========================
# METRICS
# ==========================
def compute_global_metrics(S_true, S_pred, Mask, T_true, T_pred, P_true, P_pred):
    miss = (Mask == 0)
    miss_sum = max(1, miss.sum())
    imp_acc = float(((S_pred == S_true) & miss).sum() / miss_sum)

    obs = (Mask == 1)
    obs_sum = max(1, obs.sum())
    obs_acc = float(((S_pred == S_true) & obs).sum() / obs_sum)

    overall_acc = float((S_pred == S_true).sum() / max(1, S_pred.size))
    mae_T = float(np.mean(np.abs(T_true - T_pred)))
    acc_P = float(np.mean(P_true == P_pred))

    return {
        "Temperature_MAE": mae_T,
        "Phase_Accuracy": acc_P,
        "ImpAcc": imp_acc,
        "ObsAcc": obs_acc,
        "OverallAcc": overall_acc,
        "N_test": int(T_true.shape[0]),
    }


# ==========================
# PLOT STYLE (B/W/gray, bold 16)
# ==========================
def set_bw_plot_style():
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.2,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    })


# ==========================
# TRAIN + EVAL ONE VARIANT
# ==========================
def run_one_variant(
    vname, variant_dir, device,
    train_loader, val_loader, test_loader,
    model_key, Cin, C,
    energy_per_site, nn_corr, c2_corr, mag_abs,
    # weights
    LAM_REC, LAM_E, LAM_M, LAM_C2, LAM_T, LAM_PHASE, LAM_BIN
):
    os.makedirs(variant_dir, exist_ok=True)

    net = build_model(model_key, Cin=Cin, C=C).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    ce = nn.CrossEntropyLoss()

    best_val = float("inf")
    patience = 0
    epochs_completed = 0
    t_train0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        epochs_completed = epoch
        net.train()
        train_loss = 0.0

        for batch in train_loader:
            Xb = batch["X"].to(device, non_blocking=True)
            S  = batch["S"].to(device, non_blocking=True)
            Tt = batch["T"].to(device, non_blocking=True)
            P  = batch["P"].to(device, non_blocking=True)
            Yb = batch["Y"].to(device, non_blocking=True)
            M  = batch["M"].to(device, non_blocking=True)

            S_hat, T_hat, P_log, _ = net(Xb)

            # Reconstruction
            observed_rec = (M * (S_hat - Yb).abs()).sum() / (M.sum() + 1e-8)
            missing_rec  = ((1 - M) * (S_hat - S).abs()).sum() / ((1 - M).sum() + 1e-8)
            rec = observed_rec + 2.0 * missing_rec

            # Physics
            L_Et  = (energy_per_site(S_hat).mean() - energy_per_site(S).mean().detach()).pow(2)
            L_Mt  = (mag_abs(S_hat).mean() - mag_abs(S).mean().detach()).pow(2)
            L_C2t = (c2_corr(S_hat).mean() - c2_corr(S).mean().detach()).pow(2)

            # Heads
            L_Tt = (T_hat - Tt).abs().mean()
            L_Ph = ce(P_log, P)

            # Bin
            L_bin = binary_consistency_loss(S_hat)

            loss = (LAM_REC * rec +
                    LAM_E * L_Et + LAM_M * L_Mt + LAM_C2 * L_C2t +
                    LAM_T * L_Tt + LAM_PHASE * L_Ph +
                    LAM_BIN * L_bin)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            train_loss += float(loss.item())

        train_loss /= max(1, len(train_loader))

        # Validation
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                Xb = batch["X"].to(device)
                S  = batch["S"].to(device)
                Tt = batch["T"].to(device)
                P  = batch["P"].to(device)
                Yb = batch["Y"].to(device)
                M  = batch["M"].to(device)

                S_hat, T_hat, P_log, _ = net(Xb)

                observed_rec = (M * (S_hat - Yb).abs()).sum() / (M.sum() + 1e-8)
                missing_rec  = ((1 - M) * (S_hat - S).abs()).sum() / ((1 - M).sum() + 1e-8)
                rec = observed_rec + 2.0 * missing_rec

                L_Et  = (energy_per_site(S_hat).mean() - energy_per_site(S).mean()).pow(2)
                L_Mt  = (mag_abs(S_hat).mean() - mag_abs(S).mean()).pow(2)
                L_C2t = (c2_corr(S_hat).mean() - c2_corr(S).mean()).pow(2)
                L_Tt = (T_hat - Tt).abs().mean()
                L_Ph = ce(P_log, P)
                L_bin = binary_consistency_loss(S_hat)

                batch_val = (LAM_REC * rec +
                             LAM_E * L_Et + LAM_M * L_Mt + LAM_C2 * L_C2t +
                             LAM_T * L_Tt + LAM_PHASE * L_Ph +
                             LAM_BIN * L_bin)

                val_loss += float(batch_val.item())

        val_loss /= max(1, len(val_loader))
        scheduler.step()

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            patience = 0
            torch.save({"epoch": epoch,
                        "model_state_dict": net.state_dict(),
                        "val_loss": best_val},
                       os.path.join(variant_dir, "best_model.pth"))
        else:
            patience += 1

        print(f"[{vname}] Ep {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f} | Pat {patience}/{PATIENCE}")

        if patience >= PATIENCE:
            print(f"[{vname}] ? Early stopping at epoch {epoch}")
            break

    train_time = (time.time() - t_train0) / 60.0

    # Load best
    ckpt = torch.load(os.path.join(variant_dir, "best_model.pth"), map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    final_val_loss = float(ckpt["val_loss"])

    # Test
    net.eval()
    all_Tt, all_Tp, all_Pt, all_Pp = [], [], [], []
    all_S_true, all_S_pred, all_M = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            Xb = batch["X"].to(device)
            S  = batch["S"].to(device)
            Tt = batch["T"].to(device)
            P  = batch["P"].to(device)
            M  = batch["M"].to(device)

            S_hat, T_hat, P_log, _ = net(Xb)
            S_sign = torch.where(S_hat > 0, 1.0, -1.0).float()

            all_Tt.append(Tt.cpu().numpy())
            all_Tp.append(T_hat.cpu().numpy())
            all_Pt.append(P.cpu().numpy())
            all_Pp.append(P_log.argmax(1).cpu().numpy())
            all_S_true.append(S.cpu().numpy())
            all_S_pred.append(S_sign.cpu().numpy())
            all_M.append(M.cpu().numpy())

    T_true = np.concatenate(all_Tt).reshape(-1)
    T_pred = np.concatenate(all_Tp).reshape(-1)
    P_true = np.concatenate(all_Pt).reshape(-1)
    P_pred = np.concatenate(all_Pp).reshape(-1)
    S_true = np.concatenate(all_S_true)  # (N,1,L,L)
    S_pred = np.concatenate(all_S_pred)
    Mask   = np.concatenate(all_M)

    S_true = np.where(S_true > 0, 1.0, -1.0)
    S_pred = np.where(S_pred > 0, 1.0, -1.0)

    metrics = compute_global_metrics(S_true, S_pred, Mask, T_true, T_pred, P_true, P_pred)

    # Save per-variant files
    pd.DataFrame([{
        **metrics,
        "Final_Val_Loss": final_val_loss,
        "Epochs": epochs_completed,
        "TrainTime_min": train_time,
        "ModelKey": model_key,
        "Cin": Cin,
        "Cwidth": C,
        "Weights": f"REC={LAM_REC},E={LAM_E},M={LAM_M},C2={LAM_C2},T={LAM_T},PH={LAM_PHASE},BIN={LAM_BIN}"
    }]).to_csv(os.path.join(variant_dir, "metrics_global.csv"), index=False)

    pd.DataFrame([{
        "epochs_completed": epochs_completed,
        "final_val_loss": final_val_loss,
        "training_time_minutes": train_time,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }]).to_csv(os.path.join(variant_dir, "training_times.csv"), index=False)

    return {
        "Variant": vname,
        **metrics,
        "Epochs": epochs_completed,
        "Final_Val_Loss": final_val_loss,
        "TrainTime_min": train_time,
        "ModelKey": model_key,
        "Cin": Cin,
        "Cwidth": C,
    }


# ==========================
# ABLATION VARIANTS (ONLY requested)
# ==========================
def get_variants():
    variants = []

    # 0) Reference (FULL)
    variants.append(dict(
        name="FULL",
        model_key="FULL",
        use_mask=True,
        noise_mode="file",   # use NOISY_CSV
        missing_rate=None,
        C=64,
        weights=dict(REC=BASE_LAM_REC, E=BASE_LAM_E, M=BASE_LAM_M, C2=BASE_LAM_C2, T=BASE_LAM_T, PH=BASE_LAM_PHASE, BIN=BASE_LAM_BIN)
    ))

    # 1) Attention removal
    variants.append(dict(
        name="NO_ATTENTION",
        model_key="NO_ATT",
        use_mask=True,
        noise_mode="file",
        missing_rate=None,
        C=64,
        weights=dict(REC=BASE_LAM_REC, E=BASE_LAM_E, M=BASE_LAM_M, C2=BASE_LAM_C2, T=BASE_LAM_T, PH=BASE_LAM_PHASE, BIN=BASE_LAM_BIN)
    ))

    # 2) Simpler encoder/decoder (3 blocks)
    variants.append(dict(
        name="SIMPLE3",
        model_key="SIMPLE3",
        use_mask=True,
        noise_mode="file",
        missing_rate=None,
        C=64,
        weights=dict(REC=BASE_LAM_REC, E=BASE_LAM_E, M=BASE_LAM_M, C2=BASE_LAM_C2, T=BASE_LAM_T, PH=BASE_LAM_PHASE, BIN=BASE_LAM_BIN)
    ))

    # 3) Vanilla CNN baseline
    variants.append(dict(
        name="VANILLA_CNN",
        model_key="VANILLA",
        use_mask=True,
        noise_mode="file",
        missing_rate=None,
        C=64,
        weights=dict(REC=BASE_LAM_REC, E=BASE_LAM_E, M=BASE_LAM_M, C2=BASE_LAM_C2, T=BASE_LAM_T, PH=BASE_LAM_PHASE, BIN=BASE_LAM_BIN)
    ))

    # 4) Single physics loss tests (E only, M only, C2 only) + keep REC/T/PH/BIN
    variants += [
        dict(name="PHYS_E_ONLY", model_key="FULL", use_mask=True, noise_mode="file", missing_rate=None, C=64,
             weights=dict(REC=BASE_LAM_REC, E=BASE_LAM_E, M=0.0, C2=0.0, T=BASE_LAM_T, PH=BASE_LAM_PHASE, BIN=BASE_LAM_BIN)),
        dict(name="PHYS_M_ONLY", model_key="FULL", use_mask=True, noise_mode="file", missing_rate=None, C=64,
             weights=dict(REC=BASE_LAM_REC, E=0.0, M=BASE_LAM_M, C2=0.0, T=BASE_LAM_T, PH=BASE_LAM_PHASE, BIN=BASE_LAM_BIN)),
        dict(name="PHYS_C2_ONLY", model_key="FULL", use_mask=True, noise_mode="file", missing_rate=None, C=64,
             weights=dict(REC=BASE_LAM_REC, E=0.0, M=0.0, C2=BASE_LAM_C2, T=BASE_LAM_T, PH=BASE_LAM_PHASE, BIN=BASE_LAM_BIN)),
    ]

    # 5) Weight tuning removal -> all weights = 1 (including BIN)
    variants.append(dict(
        name="NO_WEIGHT_TUNING",
        model_key="FULL",
        use_mask=True,
        noise_mode="file",
        missing_rate=None,
        C=64,
        weights=dict(REC=1.0, E=1.0, M=1.0, C2=1.0, T=1.0, PH=1.0, BIN=1.0)
    ))

    # 6) Mask channel removal (Cin=1), still use model (FULL) but without mask copying rule
    variants.append(dict(
        name="NO_MASK_CHANNEL",
        model_key="FULL",
        use_mask=False,
        noise_mode="file",
        missing_rate=None,
        C=64,
        weights=dict(REC=BASE_LAM_REC, E=BASE_LAM_E, M=BASE_LAM_M, C2=BASE_LAM_C2, T=BASE_LAM_T, PH=BASE_LAM_PHASE, BIN=BASE_LAM_BIN)
    ))

    # 7) Noise level ablation (generate noise, not file)
    for r in NOISE_LEVELS:
        variants.append(dict(
            name=f"NOISE_{int(r*100):02d}",
            model_key="FULL",
            use_mask=True,
            noise_mode="gen",
            missing_rate=r,
            C=64,
            weights=dict(REC=BASE_LAM_REC, E=BASE_LAM_E, M=BASE_LAM_M, C2=BASE_LAM_C2, T=BASE_LAM_T, PH=BASE_LAM_PHASE, BIN=BASE_LAM_BIN)
        ))

    # 8) Channel width reduction (64 -> 32)
    variants.append(dict(
        name="CWIDTH_32",
        model_key="FULL",
        use_mask=True,
        noise_mode="file",
        missing_rate=None,
        C=32,
        weights=dict(REC=BASE_LAM_REC, E=BASE_LAM_E, M=BASE_LAM_M, C2=BASE_LAM_C2, T=BASE_LAM_T, PH=BASE_LAM_PHASE, BIN=BASE_LAM_BIN)
    ))

    # 9) Channel width reduction (64 -> 16)
    variants.append(dict(
        name="CWIDTH_16",
        model_key="FULL",
        use_mask=True,
        noise_mode="file",
        missing_rate=None,
        C=16,
        weights=dict(REC=BASE_LAM_REC, E=BASE_LAM_E, M=BASE_LAM_M, C2=BASE_LAM_C2, T=BASE_LAM_T, PH=BASE_LAM_PHASE, BIN=BASE_LAM_BIN)
    ))

    return variants


# ==========================
# SUMMARY PDF (plots)
# ==========================
def plot_ablation_summary(df, out_pdf):
    set_bw_plot_style()
    x = np.arange(len(df))

    with PdfPages(out_pdf) as pdf:
        # Page 1
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, df["Temperature_MAE"], marker="o", color="black", label="Temp MAE")
        ax.plot(x, 1.0 - df["ImpAcc"], marker="s", color="dimgray", label="1 - ImpAcc")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Variant"].tolist(), rotation=45, ha="right", fontweight="bold")
        ax.set_ylabel("Error / (1-Accuracy)", fontweight="bold")
        ax.set_title("Ablation Summary (Errors)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Page 2
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, df["Phase_Accuracy"], marker="o", color="black", label="Phase Acc")
        ax.plot(x, df["OverallAcc"], marker="^", color="gray", label="Overall Acc")
        ax.plot(x, df["ObsAcc"], marker="s", color="dimgray", label="Obs Acc")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Variant"].tolist(), rotation=45, ha="right", fontweight="bold")
        ax.set_ylabel("Accuracy", fontweight="bold")
        ax.set_ylim(0, 1.01)
        ax.set_title("Ablation Summary (Accuracies)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Page 3
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, df["TrainTime_min"], marker="o", color="black", label="TrainTime (min)")
        ax.plot(x, df["Epochs"], marker="s", color="dimgray", label="Epochs")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Variant"].tolist(), rotation=45, ha="right", fontweight="bold")
        ax.set_ylabel("Minutes / Epochs", fontweight="bold")
        ax.set_title("Ablation Summary (Compute)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)


# ==========================
# MAIN
# ==========================
def run_ablation_study():
    os.makedirs(ABLATION_DIR, exist_ok=True)

    device = setup_device(DEVICE)
    energy_per_site, nn_corr, c2_corr, mag_abs = make_physics_helpers(device)

    # Load clean + file-noisy
    Sigma, Y_file, T_arr, Ph_arr = load_clean_noisy(CLEAN_CSV, NOISY_CSV, L, max_samples=DEBUG_MAX_SAMPLES)

    variants = get_variants()
    summary_rows = []

    # Save config
    with open(os.path.join(ABLATION_DIR, "ablation_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "L": L, "Tc": Tc,
            "CLEAN_CSV": CLEAN_CSV, "NOISY_CSV": NOISY_CSV,
            "EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE, "LR": LR,
            "DEVICE": DEVICE, "PATIENCE": PATIENCE, "SEED": SEED,
            "NOISE_LEVELS": NOISE_LEVELS
        }, f, indent=2)

    for v in variants:
        vname = v["name"]
        print("\n" + "="*70)
        print(f"?? Variant: {vname}")
        print("="*70)

        # Choose noisy source
        if v["noise_mode"] == "file":
            Y = Y_file.copy()
            Mask = (Y != 0).astype(np.float32)
        else:
            # generate noise at specific missing rate
            Y, Mask = make_noisy_from_clean(Sigma, v["missing_rate"], seed=SEED)

        X = build_input(Y, Mask, use_mask_channel=v["use_mask"])
        Cin = X.shape[1]

        ds = IsingDS(X, Sigma, T_arr, Ph_arr, Y, Mask)

        N = len(ds)
        n_train = int(TRAIN_FRAC * N)
        n_val = int(VAL_FRAC * N)
        n_test = N - n_train - n_val

        g = torch.Generator().manual_seed(SEED)
        train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=g)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
        test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        w = v["weights"]
        variant_dir = os.path.join(ABLATION_DIR, f"variant_{vname}")

        row = run_one_variant(
            vname=vname,
            variant_dir=variant_dir,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model_key=v["model_key"],
            Cin=Cin,
            C=v["C"],
            energy_per_site=energy_per_site,
            nn_corr=nn_corr,
            c2_corr=c2_corr,
            mag_abs=mag_abs,
            LAM_REC=w["REC"], LAM_E=w["E"], LAM_M=w["M"], LAM_C2=w["C2"],
            LAM_T=w["T"], LAM_PHASE=w["PH"], LAM_BIN=w["BIN"]
        )
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(ABLATION_DIR, "ablation_results.csv")
    df.to_csv(out_csv, index=False)

    out_pdf = os.path.join(ABLATION_DIR, "ablation_summary.pdf")
    plot_ablation_summary(df, out_pdf)

    print("\n? Ablation completed.")
    print(f"?? CSV : {out_csv}")
    print(f"?? PDF : {out_pdf}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_ablation_study()
