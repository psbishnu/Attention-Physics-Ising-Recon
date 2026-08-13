#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal physics-loss ablation for the attention-guided CNN Ising reconstructor.

Purpose
-------
1. Full
2. No_Energy
3. No_Magnetization
4. No_Correlation
5. No_Binary
6. Physics_x0.5
7. Physics_x2.0

The network architecture, input channels, data split, optimizer, seed, and
auxiliary temperature/phase losses are kept fixed. Only the four
physics-related coefficients are changed, with C2 used as the independent
correlation regularizer and C1 retained for evaluation only.

No Restormer is used.

Run separately for each lattice size by changing only:
    DATASET_NAME = "MCD32"
to MCD64, MCD128, or another MCD{L} name.

Required files
--------------
J5Data/MCD{L}.csv
J5Data/MCDN{L}.csv

The script is intentionally self-contained and needs no supporting Python file.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import gc
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


# =============================================================================
# USER SETTINGS: EDIT ONLY THIS SECTION
# =============================================================================
# ==========================================================
# DATASET SELECTION
# ==========================================================

DATASET_NAME = "MCD128"      # MCD32, MCD64, MCD128

L = int(DATASET_NAME.replace("MCD", ""))

# Dataset paths (relative to the script inside scripta/)
CLEAN_CSV = Path(f"../JOB5_Noise/J5Data/{DATASET_NAME}.csv")
NOISY_CSV = Path(f"../JOB5_Noise/J5Data/MCDN{L}.csv")
OUT_DIR   = Path(f"./Reviewer5_Ablation_{DATASET_NAME}")

# "auto" uses CUDA when available and CPU otherwise.
DEVICE = "auto"
USE_AMP = True
CPU_THREADS = 16
NUM_WORKERS = 0

EPOCHS = 40
PATIENCE = 7
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
SEED = 123

# Automatic batch sizes: L=32 -> 64, L=64 -> 16, L=128 -> 4.
# Set a positive integer here to override automatic selection.
BATCH_SIZE_OVERRIDE: Optional[int] = None

# Main objective coefficients from the reference implementation.
LAMBDA_REC = 2.0
LAMBDA_T = 1.0
LAMBDA_PHASE = 1.0
LAMBDA_MISSING = 2.0

BASE_LAMBDA_E = 0.1
BASE_LAMBDA_M = 0.1
BASE_LAMBDA_C2 = 0.1
BASE_LAMBDA_BIN = 0.5

# Minimal reviewer-requested experiments.
EXPERIMENTS_TO_RUN = [
    "Full",
    "No_Energy",
    "No_Magnetization",
    "No_Correlation",
    "No_Binary",
    "Physics_x0.5",
    "Physics_x2.0",
]

SAVE_CHECKPOINTS = True
# =============================================================================


def infer_lattice_size(dataset_name: str) -> int:
    match = re.search(r"(\d+)$", dataset_name)
    if match is None:
        raise ValueError(
            f"Cannot infer lattice size from DATASET_NAME={dataset_name!r}. "
            "Use a name ending in the lattice size, such as MCD32."
        )
    lattice_size = int(match.group(1))
    if lattice_size <= 0:
        raise ValueError("Lattice size must be positive.")
    return lattice_size


L = infer_lattice_size(DATASET_NAME)
BATCH_SIZE = (
    int(BATCH_SIZE_OVERRIDE)
    if BATCH_SIZE_OVERRIDE is not None
    else max(1, min(64, 65536 // (L * L)))
)


@dataclass(frozen=True)
class LossWeights:
    energy: float
    magnetization: float
    correlation: float  # C2 correlation weight
    binary: float


EXPERIMENTS: Mapping[str, LossWeights] = {
    "Full": LossWeights(
        BASE_LAMBDA_E, BASE_LAMBDA_M, BASE_LAMBDA_C2, BASE_LAMBDA_BIN
    ),
    "No_Energy": LossWeights(
        0.0, BASE_LAMBDA_M, BASE_LAMBDA_C2, BASE_LAMBDA_BIN
    ),
    "No_Magnetization": LossWeights(
        BASE_LAMBDA_E, 0.0, BASE_LAMBDA_C2, BASE_LAMBDA_BIN
    ),
    "No_Correlation": LossWeights(
        BASE_LAMBDA_E, BASE_LAMBDA_M, 0.0, BASE_LAMBDA_BIN
    ),
    "No_Binary": LossWeights(
        BASE_LAMBDA_E, BASE_LAMBDA_M, BASE_LAMBDA_C2, 0.0
    ),
    "Physics_x0.5": LossWeights(
        0.5 * BASE_LAMBDA_E,
        0.5 * BASE_LAMBDA_M,
        0.5 * BASE_LAMBDA_C2,
        0.5 * BASE_LAMBDA_BIN,
    ),
    "Physics_x2.0": LossWeights(
        2.0 * BASE_LAMBDA_E,
        2.0 * BASE_LAMBDA_M,
        2.0 * BASE_LAMBDA_C2,
        2.0 * BASE_LAMBDA_BIN,
    ),
}


# =============================================================================
# Reproducibility and device
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if CPU_THREADS > 0:
        torch.set_num_threads(CPU_THREADS)
        try:
            torch.set_num_interop_threads(max(1, min(4, CPU_THREADS)))
        except RuntimeError:
            pass

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    requested = DEVICE.lower().strip()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("DEVICE must be 'auto', 'cpu', or 'cuda'.")

    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    if requested == "cuda":
        print("CUDA requested but unavailable; falling back to CPU.")

    print("Using CPU")
    return torch.device("cpu")


def autocast_context(device: torch.device):
    enabled = USE_AMP and device.type == "cuda"
    if not enabled:
        return contextlib.nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)

    return torch.cuda.amp.autocast(dtype=torch.float16)


def make_grad_scaler(device: torch.device):
    enabled = USE_AMP and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# =============================================================================
# Data
# =============================================================================
def extract_spin_columns(df: pd.DataFrame, lattice_size: int) -> List[str]:
    columns = [
        column
        for column in df.columns
        if str(column).lower().startswith("spin")
    ]
    if not columns:
        raise ValueError(
            "No spin columns found. Expected names such as spin_0 or Spin1."
        )

    def suffix_index(name: str) -> int:
        digits = "".join(character for character in str(name) if character.isdigit())
        return int(digits) if digits else 0

    columns = sorted(columns, key=suffix_index)
    expected = lattice_size * lattice_size
    if len(columns) != expected:
        raise ValueError(
            f"Expected {expected} spin columns for L={lattice_size}, "
            f"but found {len(columns)}."
        )
    return columns


def phase_to_int(series: pd.Series) -> np.ndarray:
    if np.issubdtype(series.dtype, np.number):
        values = series.to_numpy(dtype=np.int64)
        unique = set(np.unique(values).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"Numeric phase labels must be 0/1; found {sorted(unique)}."
            )
        return values

    normalized = series.astype(str).str.strip().str.upper()
    # Preserve the reference-code convention: F=1 and P=0.
    mapped = normalized.map({"F": 1, "P": 0})
    if mapped.isna().any():
        unexpected = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"Unexpected phase labels: {unexpected}")
    return mapped.to_numpy(dtype=np.int64)


def load_data() -> Tuple[np.ndarray, ...]:
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"Clean CSV not found: {CLEAN_CSV}")
    if not NOISY_CSV.exists():
        raise FileNotFoundError(f"Noisy CSV not found: {NOISY_CSV}")

    print(f"Loading clean data: {CLEAN_CSV}")
    clean_df = pd.read_csv(CLEAN_CSV)
    print(f"Loading noisy data: {NOISY_CSV}")
    noisy_df = pd.read_csv(NOISY_CSV)

    if len(clean_df) != len(noisy_df):
        raise ValueError(
            f"Clean/noisy row counts differ: {len(clean_df)} vs {len(noisy_df)}."
        )
    if "Temperature" not in clean_df.columns:
        raise ValueError("Clean CSV has no 'Temperature' column.")
    if "Phase" not in clean_df.columns:
        raise ValueError("Clean CSV has no 'Phase' column.")

    spin_columns = extract_spin_columns(clean_df, L)
    missing_in_noisy = [
        column for column in spin_columns if column not in noisy_df.columns
    ]
    if missing_in_noisy:
        raise ValueError(
            f"Noisy CSV is missing {len(missing_in_noisy)} spin columns."
        )

    clean = (
        clean_df[spin_columns]
        .to_numpy(dtype=np.float32)
        .reshape(-1, L, L)
    )
    noisy = (
        noisy_df[spin_columns]
        .to_numpy(dtype=np.float32)
        .reshape(-1, L, L)
    )
    temperatures = clean_df["Temperature"].to_numpy(dtype=np.float32)
    phases = phase_to_int(clean_df["Phase"])

    # The reference code treats zero-valued sites as missing.
    mask = (noisy != 0).astype(np.float32)
    inputs = np.stack((noisy, mask), axis=1).astype(np.float32)

    return inputs, clean, temperatures, phases, noisy, mask


class IsingDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        clean: np.ndarray,
        temperatures: np.ndarray,
        phases: np.ndarray,
        noisy: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        self.inputs = torch.from_numpy(inputs)
        self.clean = torch.from_numpy(clean)[:, None]
        self.temperatures = torch.from_numpy(temperatures)[:, None]
        self.phases = torch.from_numpy(phases)
        self.noisy = torch.from_numpy(noisy)[:, None]
        self.mask = torch.from_numpy(mask)[:, None]

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "X": self.inputs[index],
            "S": self.clean[index],
            "T": self.temperatures[index],
            "P": self.phases[index],
            "Y": self.noisy[index],
            "M": self.mask[index],
        }


def deterministic_split_indices(
    number_samples: int, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(number_samples, generator=generator).tolist()

    number_train = int(0.8 * number_samples)
    number_validation = int(0.1 * number_samples)

    train_indices = permutation[:number_train]
    validation_indices = permutation[
        number_train:number_train + number_validation
    ]
    test_indices = permutation[number_train + number_validation:]
    return train_indices, validation_indices, test_indices


def make_loaders(
    dataset: Dataset,
    split_indices: Tuple[List[int], List[int], List[int]],
    device: torch.device,
    experiment_seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_indices, validation_indices, test_indices = split_indices
    pin_memory = device.type == "cuda"
    persistent_workers = NUM_WORKERS > 0

    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        shuffle=True,
        generator=torch.Generator().manual_seed(experiment_seed),
        **common,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader, test_loader


# =============================================================================
# Reference CNN architecture (no Restormer)
# =============================================================================
class PhysicsGuidedCNN(nn.Module):
    """Residual encoder-decoder with the reference spatial-attention module."""

    def __init__(self) -> None:
        super().__init__()
        channels = 64

        self.enc_conv1 = nn.Conv2d(
            2, channels, 3, padding=1, padding_mode="circular"
        )
        self.enc_conv2 = nn.Conv2d(
            channels, channels, 3, padding=1, padding_mode="circular"
        )
        self.enc_conv3 = nn.Conv2d(
            channels, channels, 3, padding=1, padding_mode="circular"
        )
        self.enc_conv4 = nn.Conv2d(
            channels, channels, 3, padding=1, padding_mode="circular"
        )
        self.enc_conv5 = nn.Conv2d(
            channels, channels, 3, padding=1, padding_mode="circular"
        )

        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )

        self.dec_conv1 = nn.Conv2d(
            channels + 1, 64, 3, padding=1, padding_mode="circular"
        )
        self.dec_conv2 = nn.Conv2d(
            64, 64, 3, padding=1, padding_mode="circular"
        )
        self.dec_conv3 = nn.Conv2d(
            64, 32, 3, padding=1, padding_mode="circular"
        )
        self.dec_conv4 = nn.Conv2d(
            32, 16, 3, padding=1, padding_mode="circular"
        )
        self.dec_conv5 = nn.Conv2d(16, 1, 1)

        self.temperature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.phase_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noisy_input = x[:, :1]
        mask = x[:, 1:2]

        h1 = self.activation(self.enc_conv1(x))
        h2 = self.activation(self.enc_conv2(h1)) + h1
        h3 = self.activation(self.enc_conv3(h2)) + h2
        h4 = self.activation(self.enc_conv4(h3)) + h3
        h5 = self.activation(self.enc_conv5(h4)) + h4

        attention_map = self.attention(h5)
        enhanced_features = torch.cat((h5, attention_map), dim=1)

        d1 = self.activation(self.dec_conv1(enhanced_features))
        d2 = self.activation(self.dec_conv2(d1)) + d1
        d3 = self.activation(self.dec_conv3(d2))
        d4 = self.activation(self.dec_conv4(d3))
        raw_spin = torch.tanh(self.dec_conv5(d4))

        # Preserve observed sites exactly and reconstruct only missing sites.
        final_spin = mask * noisy_input + (1.0 - mask) * raw_spin

        temperature = self.temperature_head(h5)
        phase_logits = self.phase_head(h5)
        return final_spin, temperature, phase_logits


# =============================================================================
# Differentiable physics
# =============================================================================
def energy_per_site_soft(spins: torch.Tensor) -> torch.Tensor:
    """
    Differentiable 2D Ising nearest-neighbor energy per site (J=1).

    Each right/down bond is counted once:
        e = -mean(s_ij s_i,j+1 + s_ij s_i+1,j)
    """
    right = torch.roll(spins, shifts=-1, dims=3)
    down = torch.roll(spins, shifts=-1, dims=2)
    return -(
        spins * right + spins * down
    ).mean(dim=(1, 2, 3))


def nearest_neighbor_correlation_soft(spins: torch.Tensor) -> torch.Tensor:
    """
    Differentiable nearest-neighbor correlation C1.

    C1 is retained as an evaluation observable only. For the nearest-neighbor
    Ising model with J=1 and the energy definition used here, e = -2*C1, so C1
    is not used as an independent training regularizer.
    """
    right = torch.roll(spins, shifts=-1, dims=3)
    down = torch.roll(spins, shifts=-1, dims=2)
    return 0.5 * (
        spins * right + spins * down
    ).mean(dim=(1, 2, 3))


def second_neighbor_correlation_soft(spins: torch.Tensor) -> torch.Tensor:
    """
    Differentiable second-neighbor correlation C2 at axial separation two
    lattice spacings with periodic boundary conditions.
    """
    right2 = torch.roll(spins, shifts=-2, dims=3)
    left2 = torch.roll(spins, shifts=2, dims=3)
    down2 = torch.roll(spins, shifts=-2, dims=2)
    up2 = torch.roll(spins, shifts=2, dims=2)
    return 0.25 * (
        spins * right2
        + spins * left2
        + spins * down2
        + spins * up2
    ).mean(dim=(1, 2, 3))


def absolute_magnetization_soft(spins: torch.Tensor) -> torch.Tensor:
    return spins.mean(dim=(1, 2, 3)).abs()


def binary_consistency_loss(spins: torch.Tensor) -> torch.Tensor:
    return ((spins.abs() - 1.0) ** 2).mean()


def hard_spins(spins: torch.Tensor) -> torch.Tensor:
    return torch.where(spins >= 0.0, 1.0, -1.0)


def physics_loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    # Per-sample matching is stronger and more stable than matching only
    # the batch-averaged observable.
    energy_loss = F.mse_loss(
        energy_per_site_soft(prediction),
        energy_per_site_soft(target),
    )
    magnetization_loss = F.mse_loss(
        absolute_magnetization_soft(prediction),
        absolute_magnetization_soft(target),
    )
    correlation_loss = F.mse_loss(
        second_neighbor_correlation_soft(prediction),
        second_neighbor_correlation_soft(target),
    )
    binary_loss = binary_consistency_loss(prediction)

    return {
        "energy": energy_loss,
        "magnetization": magnetization_loss,
        "correlation": correlation_loss,
        "binary": binary_loss,
    }


# =============================================================================
# Training and validation
# =============================================================================
def move_batch(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def compute_loss(
    model_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: LossWeights,
    phase_criterion: nn.Module,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    spin_hat, temperature_hat, phase_logits = model_outputs

    clean = batch["S"]
    target_temperature = batch["T"]
    target_phase = batch["P"]
    noisy = batch["Y"]
    mask = batch["M"]

    observed_loss = (
        mask * torch.abs(spin_hat - noisy)
    ).sum() / (mask.sum() + 1e-8)

    missing_mask = 1.0 - mask
    missing_loss = (
        missing_mask * torch.abs(spin_hat - clean)
    ).sum() / (missing_mask.sum() + 1e-8)

    reconstruction_loss = observed_loss + LAMBDA_MISSING * missing_loss
    physical = physics_loss_components(spin_hat, clean)
    temperature_loss = torch.abs(
        temperature_hat - target_temperature
    ).mean()
    phase_loss = phase_criterion(phase_logits, target_phase)

    total_loss = (
        LAMBDA_REC * reconstruction_loss
        + weights.energy * physical["energy"]
        + weights.magnetization * physical["magnetization"]
        + weights.correlation * physical["correlation"]
        + weights.binary * physical["binary"]
        + LAMBDA_T * temperature_loss
        + LAMBDA_PHASE * phase_loss
    )

    components = {
        "total": total_loss,
        "reconstruction": reconstruction_loss,
        "observed": observed_loss,
        "missing": missing_loss,
        "energy": physical["energy"],
        "magnetization": physical["magnetization"],
        "correlation": physical["correlation"],
        "binary": physical["binary"],
        "temperature": temperature_loss,
        "phase": phase_loss,
    }
    return total_loss, components


def empty_meter() -> Dict[str, float]:
    return {
        "total": 0.0,
        "reconstruction": 0.0,
        "observed": 0.0,
        "missing": 0.0,
        "energy": 0.0,
        "magnetization": 0.0,
        "correlation": 0.0,
        "binary": 0.0,
        "temperature": 0.0,
        "phase": 0.0,
    }


def update_meter(
    meter: Dict[str, float], components: Dict[str, torch.Tensor]
) -> None:
    for key in meter:
        meter[key] += float(components[key].detach().item())


def average_meter(meter: Dict[str, float], count: int) -> Dict[str, float]:
    divisor = max(1, count)
    return {key: value / divisor for key, value in meter.items()}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    weights: LossWeights,
    phase_criterion: nn.Module,
    epoch: int,
    experiment_name: str,
) -> Dict[str, float]:
    model.train()
    meter = empty_meter()

    progress = tqdm(
        loader,
        desc=f"{experiment_name} | epoch {epoch:03d} train",
        leave=False,
        ncols=110,
    )

    for batch in progress:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device):
            outputs = model(batch["X"])
            total_loss, components = compute_loss(
                outputs, batch, weights, phase_criterion
            )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        update_meter(meter, components)
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                total=f"{components['total'].item():.4f}",
                rec=f"{components['reconstruction'].item():.4f}",
            )

    return average_meter(meter, len(loader))


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: LossWeights,
    phase_criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    meter = empty_meter()

    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device):
            outputs = model(batch["X"])
            _, components = compute_loss(
                outputs, batch, weights, phase_criterion
            )
        update_meter(meter, components)

    return average_meter(meter, len(loader))


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_one_experiment(
    experiment_name: str,
    weights: LossWeights,
    dataset: Dataset,
    split_indices: Tuple[List[int], List[int], List[int]],
    device: torch.device,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, float]]:
    # Reset the seed so every ablation begins with identical initialization.
    set_seed(SEED)

    train_loader, validation_loader, _ = make_loaders(
        dataset, split_indices, device, experiment_seed=SEED
    )

    model = PhysicsGuidedCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    phase_criterion = nn.CrossEntropyLoss()
    scaler = make_grad_scaler(device)

    experiment_dir = OUT_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = experiment_dir / "best_model.pth"

    best_selection_loss = math.inf
    best_total_loss = math.inf
    best_epoch = 0
    patience_counter = 0
    history_rows: List[Dict[str, float]] = []
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            weights,
            phase_criterion,
            epoch,
            experiment_name,
        )
        validation_metrics = validate_epoch(
            model,
            validation_loader,
            device,
            weights,
            phase_criterion,
        )
        scheduler.step()

        # Common early-stopping criterion for fair comparison across coefficients.
        selection_loss = validation_metrics["reconstruction"]
        improved = selection_loss < best_selection_loss

        if improved:
            best_selection_loss = selection_loss
            best_total_loss = validation_metrics["total"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_reconstruction_loss": best_selection_loss,
                    "validation_total_loss": best_total_loss,
                    "experiment": experiment_name,
                    "weights": asdict(weights),
                    "dataset": DATASET_NAME,
                    "L": L,
                    "seed": SEED,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        row: Dict[str, float] = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - epoch_start,
        }
        row.update({
            f"train_{key}": value for key, value in train_metrics.items()
        })
        row.update({
            f"val_{key}": value for key, value in validation_metrics.items()
        })
        history_rows.append(row)

        marker = "*" if improved else " "
        print(
            f"{marker} {experiment_name:18s} | "
            f"epoch {epoch:03d}/{EPOCHS} | "
            f"train={train_metrics['total']:.5f} | "
            f"val={validation_metrics['total']:.5f} | "
            f"val_rec={validation_metrics['reconstruction']:.5f} | "
            f"patience={patience_counter}/{PATIENCE}"
        )

        if patience_counter >= PATIENCE:
            print(f"Early stopping: {experiment_name} at epoch {epoch}.")
            break

    training_seconds = time.time() - start
    history = pd.DataFrame(history_rows)
    history.to_csv(experiment_dir / "training_history.csv", index=False)

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if not SAVE_CHECKPOINTS:
        checkpoint_path.unlink(missing_ok=True)

    training_info = {
        "Best_Epoch": int(checkpoint["epoch"]),
        "Best_Val_Reconstruction_Loss": float(
            checkpoint["validation_reconstruction_loss"]
        ),
        "Best_Val_Total_Loss": float(checkpoint["validation_total_loss"]),
        "Training_Seconds": float(training_seconds),
        "Training_Minutes": float(training_seconds / 60.0),
    }
    return model, history, training_info


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()

    temperature_absolute_error = 0.0
    phase_correct = 0
    sample_count = 0

    missing_correct = 0
    missing_count = 0
    observed_correct = 0
    observed_count = 0
    overall_correct = 0
    overall_count = 0

    energy_absolute_error = 0.0
    magnetization_absolute_error = 0.0
    c1_absolute_error = 0.0
    c2_absolute_error = 0.0

    prediction_rows: List[Dict[str, float]] = []

    for batch in tqdm(
        test_loader, desc="Test evaluation", leave=False, ncols=110
    ):
        original_indices = batch["index"].numpy()
        batch = move_batch(batch, device)

        with autocast_context(device):
            spin_hat, temperature_hat, phase_logits = model(batch["X"])

        spin_prediction = hard_spins(spin_hat)
        spin_target = hard_spins(batch["S"])
        phase_prediction = phase_logits.argmax(dim=1)

        batch_size = batch["X"].shape[0]
        sample_count += batch_size

        temperature_absolute_error += float(
            torch.abs(temperature_hat - batch["T"]).sum().item()
        )
        phase_correct += int(
            (phase_prediction == batch["P"]).sum().item()
        )

        missing = batch["M"] == 0
        observed = batch["M"] == 1
        equal = spin_prediction == spin_target

        missing_correct += int((equal & missing).sum().item())
        missing_count += int(missing.sum().item())
        observed_correct += int((equal & observed).sum().item())
        observed_count += int(observed.sum().item())
        overall_correct += int(equal.sum().item())
        overall_count += int(equal.numel())

        predicted_energy = energy_per_site_soft(spin_prediction)
        target_energy = energy_per_site_soft(spin_target)
        predicted_magnetization = absolute_magnetization_soft(spin_prediction)
        target_magnetization = absolute_magnetization_soft(spin_target)
        predicted_c1 = nearest_neighbor_correlation_soft(spin_prediction)
        target_c1 = nearest_neighbor_correlation_soft(spin_target)
        predicted_c2 = second_neighbor_correlation_soft(spin_prediction)
        target_c2 = second_neighbor_correlation_soft(spin_target)

        energy_absolute_error += float(
            torch.abs(predicted_energy - target_energy).sum().item()
        )
        magnetization_absolute_error += float(
            torch.abs(
                predicted_magnetization - target_magnetization
            ).sum().item()
        )
        c1_absolute_error += float(
            torch.abs(predicted_c1 - target_c1).sum().item()
        )
        c2_absolute_error += float(
            torch.abs(predicted_c2 - target_c2).sum().item()
        )

        true_temperature = batch["T"][:, 0].cpu().numpy()
        predicted_temperature = temperature_hat[:, 0].float().cpu().numpy()
        true_phase = batch["P"].cpu().numpy()
        predicted_phase = phase_prediction.cpu().numpy()
        e_true = target_energy.cpu().numpy()
        e_pred = predicted_energy.cpu().numpy()
        m_true = target_magnetization.cpu().numpy()
        m_pred = predicted_magnetization.cpu().numpy()
        c1_true = target_c1.cpu().numpy()
        c1_pred = predicted_c1.cpu().numpy()
        c2_true = target_c2.cpu().numpy()
        c2_pred = predicted_c2.cpu().numpy()

        for index in range(batch_size):
            prediction_rows.append(
                {
                    "DatasetIndex": int(original_indices[index]),
                    "T_true": float(true_temperature[index]),
                    "T_pred": float(predicted_temperature[index]),
                    "Phase_true": int(true_phase[index]),
                    "Phase_pred": int(predicted_phase[index]),
                    "Energy_true": float(e_true[index]),
                    "Energy_pred": float(e_pred[index]),
                    "Mabs_true": float(m_true[index]),
                    "Mabs_pred": float(m_pred[index]),
                    "C1_true": float(c1_true[index]),
                    "C1_pred": float(c1_pred[index]),
                    "C2_true": float(c2_true[index]),
                    "C2_pred": float(c2_pred[index]),
                }
            )

    if sample_count == 0:
        raise RuntimeError("Test set is empty.")
    if missing_count == 0:
        raise RuntimeError("Test set contains no missing sites.")

    metrics = {
        "Temperature_MAE": temperature_absolute_error / sample_count,
        "Phase_Accuracy": phase_correct / sample_count,
        "Imputation_Accuracy_Missing": missing_correct / missing_count,
        "Observation_Accuracy": (
            observed_correct / max(1, observed_count)
        ),
        "Overall_Reconstruction_Accuracy": (
            overall_correct / max(1, overall_count)
        ),
        "Energy_MAE": energy_absolute_error / sample_count,
        "Magnetization_MAE": magnetization_absolute_error / sample_count,
        "C1_MAE": c1_absolute_error / sample_count,
        "C2_MAE": c2_absolute_error / sample_count,
        "N_Test": sample_count,
        "N_Missing_Sites": missing_count,
    }
    return metrics, pd.DataFrame(prediction_rows)


# =============================================================================
# Output tables and figures
# =============================================================================
def add_change_from_full(summary: pd.DataFrame) -> pd.DataFrame:
    if "Full" not in set(summary["Experiment"]):
        return summary

    result = summary.copy()
    full_row = result.loc[result["Experiment"] == "Full"].iloc[0]

    higher_is_better = [
        "Phase_Accuracy",
        "Imputation_Accuracy_Missing",
        "Observation_Accuracy",
        "Overall_Reconstruction_Accuracy",
    ]
    lower_is_better = [
        "Temperature_MAE",
        "Energy_MAE",
        "Magnetization_MAE",
        "C2_MAE",
    ]

    for column in higher_is_better:
        result[f"Delta_{column}_vs_Full"] = result[column] - full_row[column]

    for column in lower_is_better:
        result[f"Delta_{column}_vs_Full"] = result[column] - full_row[column]

    return result


def save_comparison_pdf(summary: pd.DataFrame, output_path: Path) -> None:
    labels = summary["Experiment"].tolist()
    x = np.arange(len(labels))

    with PdfPages(output_path) as pdf:
        figure = plt.figure(figsize=(11, 6.5))
        plt.bar(x, summary["Imputation_Accuracy_Missing"])
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("Accuracy")
        plt.title(f"Missing-site reconstruction accuracy, L={L}")
        plt.ylim(
            max(0.0, float(summary["Imputation_Accuracy_Missing"].min()) - 0.05),
            1.0,
        )
        plt.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(11, 6.5))
        width = 0.25
        plt.bar(
            x - width,
            summary["Energy_MAE"],
            width=width,
            label="Energy MAE",
        )
        plt.bar(
            x,
            summary["Magnetization_MAE"],
            width=width,
            label="Magnetization MAE",
        )
        plt.bar(
            x + width,
            summary["C2_MAE"],
            width=width,
            label="$C_2$ MAE",
        )
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("Mean absolute error")
        plt.title(f"Physical-observable errors, L={L}")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        sensitivity = summary[
            summary["Experiment"].isin(
                ["Physics_x0.5", "Full", "Physics_x2.0"]
            )
        ].copy()
        sensitivity["PhysicsScale"] = sensitivity["Experiment"].map(
            {"Physics_x0.5": 0.5, "Full": 1.0, "Physics_x2.0": 2.0}
        )
        sensitivity = sensitivity.sort_values("PhysicsScale")

        figure = plt.figure(figsize=(8, 6))
        plt.plot(
            sensitivity["PhysicsScale"],
            sensitivity["Imputation_Accuracy_Missing"],
            marker="o",
            label="Missing-site accuracy",
        )
        plt.plot(
            sensitivity["PhysicsScale"],
            sensitivity["Overall_Reconstruction_Accuracy"],
            marker="s",
            label="Overall accuracy",
        )
        plt.xlabel("Common physics-loss coefficient scale")
        plt.ylabel("Accuracy")
        plt.title(f"Physics-loss sensitivity, L={L}")
        plt.xticks([0.5, 1.0, 2.0])
        plt.legend()
        plt.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(8, 6))
        plt.plot(
            sensitivity["PhysicsScale"],
            sensitivity["Energy_MAE"],
            marker="o",
            label="Energy MAE",
        )
        plt.plot(
            sensitivity["PhysicsScale"],
            sensitivity["Magnetization_MAE"],
            marker="s",
            label="Magnetization MAE",
        )
        plt.plot(
            sensitivity["PhysicsScale"],
            sensitivity["C2_MAE"],
            marker="^",
            label="$C_2$ MAE",
        )
        plt.xlabel("Common physics-loss coefficient scale")
        plt.ylabel("Mean absolute error")
        plt.title(f"Physical consistency sensitivity, L={L}")
        plt.xticks([0.5, 1.0, 2.0])
        plt.legend()
        plt.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)


def write_settings_file() -> None:
    settings_path = OUT_DIR / "run_settings.txt"
    with settings_path.open("w", encoding="utf-8") as handle:
        handle.write(f"DATASET_NAME={DATASET_NAME}\n")
        handle.write(f"L={L}\n")
        handle.write(f"CLEAN_CSV={CLEAN_CSV}\n")
        handle.write(f"NOISY_CSV={NOISY_CSV}\n")
        handle.write(f"DEVICE={DEVICE}\n")
        handle.write(f"BATCH_SIZE={BATCH_SIZE}\n")
        handle.write(f"EPOCHS={EPOCHS}\n")
        handle.write(f"PATIENCE={PATIENCE}\n")
        handle.write(f"LEARNING_RATE={LEARNING_RATE}\n")
        handle.write(f"WEIGHT_DECAY={WEIGHT_DECAY}\n")
        handle.write(f"SEED={SEED}\n")
        handle.write(f"LAMBDA_REC={LAMBDA_REC}\n")
        handle.write(f"LAMBDA_T={LAMBDA_T}\n")
        handle.write(f"LAMBDA_PHASE={LAMBDA_PHASE}\n")
        handle.write(f"LAMBDA_MISSING={LAMBDA_MISSING}\n")
        handle.write(f"BASE_LAMBDA_E={BASE_LAMBDA_E}\n")
        handle.write(f"BASE_LAMBDA_M={BASE_LAMBDA_M}\n")
        handle.write(f"BASE_LAMBDA_C2={BASE_LAMBDA_C2}\n")
        handle.write(f"BASE_LAMBDA_BIN={BASE_LAMBDA_BIN}\n")
        handle.write(f"EXPERIMENTS_TO_RUN={EXPERIMENTS_TO_RUN}\n")
        handle.write(
            "NOTE=C1 is retained as an evaluation observable only. For the "
            "nearest-neighbor Ising model with J=1, Energy=-2*C1, so C1 is "
            "not used as an independent training regularizer. C2 at axial "
            "separation two is used for the correlation loss and its ablation.\n"
        )
        handle.write(
            "NOTE=Training physics observables are computed from continuous "
            "tanh outputs, without torch.sign, to preserve gradients. "
            "Binarization is used only for final evaluation.\n"
        )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    unknown = [
        experiment
        for experiment in EXPERIMENTS_TO_RUN
        if experiment not in EXPERIMENTS
    ]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_settings_file()

    print("=" * 88)
    print("MINIMAL PHYSICS-LOSS ABLATION: ATTENTION-GUIDED CNN, NO RESTORMER")
    print("=" * 88)
    print(f"Dataset             : {DATASET_NAME}")
    print(f"Lattice size        : {L}")
    print(f"Clean CSV           : {CLEAN_CSV}")
    print(f"Noisy CSV           : {NOISY_CSV}")
    print(f"Output directory    : {OUT_DIR}")
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Experiments         : {len(EXPERIMENTS_TO_RUN)}")

    set_seed(SEED)
    device = resolve_device()

    arrays = load_data()
    dataset = IsingDataset(*arrays)
    split_indices = deterministic_split_indices(len(dataset), SEED)

    train_indices, validation_indices, test_indices = split_indices
    print(
        f"Samples             : total={len(dataset)}, "
        f"train={len(train_indices)}, validation={len(validation_indices)}, "
        f"test={len(test_indices)}"
    )

    summary_rows: List[Dict[str, float]] = []
    total_start = time.time()

    for experiment_number, experiment_name in enumerate(
        EXPERIMENTS_TO_RUN, start=1
    ):
        weights = EXPERIMENTS[experiment_name]
        print("\n" + "=" * 88)
        print(
            f"EXPERIMENT {experiment_number}/{len(EXPERIMENTS_TO_RUN)}: "
            f"{experiment_name}"
        )
        print(
            f"lambda_E={weights.energy}, "
            f"lambda_M={weights.magnetization}, "
            f"lambda_C2={weights.correlation}, "
            f"lambda_BIN={weights.binary}"
        )
        print("=" * 88)

        model, history, training_info = train_one_experiment(
            experiment_name,
            weights,
            dataset,
            split_indices,
            device,
        )

        _, _, test_loader = make_loaders(
            dataset, split_indices, device, experiment_seed=SEED
        )
        metrics, predictions = evaluate_model(model, test_loader, device)

        experiment_dir = OUT_DIR / experiment_name
        predictions.to_csv(
            experiment_dir / "test_predictions.csv", index=False
        )

        row: Dict[str, float] = {
            "Experiment": experiment_name,
            "Dataset": DATASET_NAME,
            "L": L,
            "Lambda_E": weights.energy,
            "Lambda_M": weights.magnetization,
            "Lambda_C2": weights.correlation,
            "Lambda_BIN": weights.binary,
            "Lambda_REC": LAMBDA_REC,
            "Lambda_T": LAMBDA_T,
            "Lambda_PHASE": LAMBDA_PHASE,
            "Seed": SEED,
            "Batch_Size": BATCH_SIZE,
            "Trainable_Parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        }
        row.update(training_info)
        row.update(metrics)
        summary_rows.append(row)

        pd.DataFrame([row]).to_csv(
            experiment_dir / "metrics.csv", index=False
        )
        partial_summary = add_change_from_full(pd.DataFrame(summary_rows))
        partial_summary.to_csv(
            OUT_DIR / f"ablation_summary_L{L}_PARTIAL.csv",
            index=False,
        )

        del model, history, test_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = add_change_from_full(pd.DataFrame(summary_rows))
    summary_path = OUT_DIR / f"ablation_summary_L{L}.csv"
    summary.to_csv(summary_path, index=False)

    reviewer_columns = [
        "Experiment",
        "Lambda_E",
        "Lambda_M",
        "Lambda_C2",
        "Lambda_BIN",
        "Temperature_MAE",
        "Phase_Accuracy",
        "Imputation_Accuracy_Missing",
        "Overall_Reconstruction_Accuracy",
        "Energy_MAE",
        "Magnetization_MAE",
        "C2_MAE",
        "C1_MAE",
        "Best_Epoch",
        "Training_Minutes",
    ]
    reviewer_table = summary[reviewer_columns].copy()
    reviewer_table.to_csv(
        OUT_DIR / f"reviewer5_minimal_table_L{L}.csv",
        index=False,
    )

    pdf_path = OUT_DIR / f"reviewer5_ablation_plots_L{L}.pdf"
    save_comparison_pdf(summary, pdf_path)

    total_seconds = time.time() - total_start
    with (OUT_DIR / "completion.txt").open("w", encoding="utf-8") as handle:
        handle.write("Status=COMPLETED\n")
        handle.write(f"Dataset={DATASET_NAME}\n")
        handle.write(f"L={L}\n")
        handle.write(f"TotalSeconds={total_seconds}\n")
        handle.write(f"TotalHours={total_seconds / 3600.0}\n")

    print("\n" + "=" * 88)
    print("ALL ABLATION EXPERIMENTS COMPLETED")
    print("=" * 88)
    print(f"Summary CSV          : {summary_path}")
    print(
        f"Reviewer table       : "
        f"{OUT_DIR / f'reviewer5_minimal_table_L{L}.csv'}"
    )
    print(f"Comparison PDF       : {pdf_path}")
    print(f"Total time           : {total_seconds / 3600.0:.2f} hours")
    print("\nReviewer-facing minimal interpretation:")
    print(
        "Use Full vs No_Energy/No_Magnetization/No_Correlation/No_Binary "
        "for individual contributions, and Physics_x0.5/Full/Physics_x2.0 "
        "for coefficient sensitivity."
    )
    print(
        "Important: C1 is evaluation-only because E=-2*C1 for J=1; "
        "the independent correlation ablation uses C2 at axial separation two."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Execution interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
