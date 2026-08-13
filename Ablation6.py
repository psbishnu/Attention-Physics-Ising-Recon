#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation6.py

Minimal robustness evaluation:
1. Spin-flip noise robustness: train at 30%, test at 10%, 30%, 50%.
2. Random masking robustness: train at 40%, test at 20%, 40%, 60%.

The model is the EnhancedInvNet CNN from the supplied J5DL notebook.
Restormer is NOT used.

Physics regularization uses energy, absolute magnetization, second-neighbor
correlation C2, and binary-spin consistency. C1 is retained for evaluation only.

Only the clean MCD{L}.csv configurations are required to create controlled
corruption levels. MCDN{L}.csv is retained as a reference path but is not used
to create the new corruption levels.

Run:
    python Ablation6.py

For another lattice size, change only:
    DATASET_NAME = "MCD64"
or:
    DATASET_NAME = "MCD128"
"""

from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# USER SETTINGS: EDIT ONLY THIS SECTION
# =============================================================================

DATASET_NAME = "MCD32"                    # MCD32, MCD64, or MCD128

# Ablation6.py is assumed to be inside the folder "scripta".
# The data are assumed to be in the sibling path:
# ../JOB5_Noise/J5Data/
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR / "../JOB5_Noise/J5Data").resolve()

L = int(DATASET_NAME.replace("MCD", ""))
CLEAN_CSV = DATA_DIR / f"MCD{L}.csv"
REFERENCE_NOISY_CSV = DATA_DIR / f"MCDN{L}.csv"

OUT_DIR = SCRIPT_DIR / f"Ablation6_{DATASET_NAME}"

DEVICE = "cuda"                           # Automatically falls back to CPU
EPOCHS = 40
PATIENCE = 7
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = {32: 64, 64: 16, 128: 2}[L]
NUM_WORKERS = 0
SEED = 123

# Minimal protocol: only two models are trained.
NOISE_TRAIN_LEVEL = 0.30
NOISE_TEST_LEVELS = (0.10, 0.30, 0.50)

MASK_TRAIN_LEVEL = 0.40
MASK_TEST_LEVELS = (0.20, 0.40, 0.60)

# Full physics-guided objective from the main framework.
LAM_REC = 2.0
LAM_E = 0.1
LAM_M = 0.1
LAM_C2 = 0.1
LAM_T = 1.0
LAM_PHASE = 1.0
LAM_BIN = 0.5

# =============================================================================


def validate_settings() -> None:
    if L not in (32, 64, 128):
        raise ValueError("DATASET_NAME must be MCD32, MCD64, or MCD128.")
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"Clean CSV not found: {CLEAN_CSV}")
    if not 0.0 < NOISE_TRAIN_LEVEL < 1.0:
        raise ValueError("NOISE_TRAIN_LEVEL must be between 0 and 1.")
    if not 0.0 < MASK_TRAIN_LEVEL < 1.0:
        raise ValueError("MASK_TRAIN_LEVEL must be between 0 and 1.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    if DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    if DEVICE == "cuda":
        print("CUDA is unavailable; using CPU.")
    else:
        print("Using CPU.")
    return torch.device("cpu")


def extract_spin_columns(columns: List[str], lattice_size: int) -> List[str]:
    spin_columns = [
        column for column in columns
        if column.lower().startswith("spin")
    ]

    def suffix_index(name: str) -> int:
        digits = "".join(character for character in name if character.isdigit())
        return int(digits) if digits else 0

    spin_columns = sorted(spin_columns, key=suffix_index)
    expected = lattice_size * lattice_size

    if len(spin_columns) != expected:
        raise ValueError(
            f"Expected {expected} spin columns for L={lattice_size}, "
            f"but found {len(spin_columns)}."
        )
    return spin_columns


def phase_to_int(series: pd.Series) -> np.ndarray:
    if np.issubdtype(series.dtype, np.number):
        values = series.to_numpy(dtype=np.int64)
        unique = set(np.unique(values).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(f"Numeric phase labels must be 0/1; found {unique}.")
        return values

    normalized = series.astype(str).str.strip().str.upper()
    mapped = normalized.map({"F": 1, "P": 0})
    if mapped.isna().any():
        bad = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"Unexpected phase labels: {bad}")
    return mapped.to_numpy(dtype=np.int64)


def load_clean_data(
    clean_path: Path,
    lattice_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"Loading clean configurations from: {clean_path}")

    header = pd.read_csv(clean_path, nrows=0)
    spin_columns = extract_spin_columns(header.columns.tolist(), lattice_size)

    dtype_map = {column: np.int8 for column in spin_columns}
    dtype_map["Temperature"] = np.float32

    dataframe = pd.read_csv(
        clean_path,
        usecols=["Temperature", "Phase", *spin_columns],
        dtype=dtype_map,
    )

    spins = (
        dataframe[spin_columns]
        .to_numpy(dtype=np.int8)
        .reshape(-1, lattice_size, lattice_size)
    )
    temperatures = dataframe["Temperature"].to_numpy(dtype=np.float32)
    phases = phase_to_int(dataframe["Phase"])

    print(f"Loaded {len(spins):,} samples for L={lattice_size}.")
    return spins, temperatures, phases


class CleanIsingDataset(Dataset):
    def __init__(
        self,
        clean_spins: np.ndarray,
        temperatures: np.ndarray,
        phases: np.ndarray,
    ) -> None:
        # Keep spins as int8 in memory; convert each batch to float32 later.
        self.clean_spins = torch.from_numpy(clean_spins)[:, None]
        self.temperatures = torch.from_numpy(temperatures)[:, None]
        self.phases = torch.from_numpy(phases)

    def __len__(self) -> int:
        return self.clean_spins.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "S": self.clean_spins[index],
            "T": self.temperatures[index],
            "P": self.phases[index],
        }


def split_indices(
    number_samples: int,
    seed: int,
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


def make_loaders(dataset: Dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_indices, validation_indices, test_indices = split_indices(
        len(dataset),
        SEED,
    )

    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available() and DEVICE == "cuda",
    }

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
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


def apply_corruption(
    clean: torch.Tensor,
    corruption_type: str,
    level: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        model_input: [corrupted spins, observation mask]
        corrupted: corrupted spin tensor
        affected: sites that were flipped or masked
    """
    random_values = torch.rand(
        clean.shape,
        device=clean.device,
        dtype=torch.float32,
    )
    affected = random_values < level

    if corruption_type == "noise":
        # Unknown spin-flip noise. The mask is all ones because no site is missing.
        corrupted = torch.where(affected, -clean, clean)
        observation_mask = torch.ones_like(clean)

    elif corruption_type == "mask":
        # Missing sites are represented by zero.
        corrupted = clean.clone()
        corrupted[affected] = 0.0
        observation_mask = (~affected).float()

    else:
        raise ValueError("corruption_type must be 'noise' or 'mask'.")

    model_input = torch.cat((corrupted, observation_mask), dim=1)
    return model_input, corrupted, affected


# =============================================================================
# EnhancedInvNet from the supplied main notebook
# =============================================================================

class EnhancedInvNet(nn.Module):
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
        self,
        model_input: torch.Tensor,
        preserve_observed: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        corrupted = model_input[:, :1]
        observation_mask = model_input[:, 1:2]

        h1 = self.activation(self.enc_conv1(model_input))
        h2 = self.activation(self.enc_conv2(h1)) + h1
        h3 = self.activation(self.enc_conv3(h2)) + h2
        h4 = self.activation(self.enc_conv4(h3)) + h3
        h5 = self.activation(self.enc_conv5(h4)) + h4

        attention_map = self.attention(h5)
        features = torch.cat((h5, attention_map), dim=1)

        d1 = self.activation(self.dec_conv1(features))
        d2 = self.activation(self.dec_conv2(d1)) + d1
        d3 = self.activation(self.dec_conv3(d2))
        d4 = self.activation(self.dec_conv4(d3))
        raw_reconstruction = torch.tanh(self.dec_conv5(d4))

        if preserve_observed:
            reconstruction = (
                observation_mask * corrupted
                + (1.0 - observation_mask) * raw_reconstruction
            )
        else:
            # Required for unknown spin-flip noise: the network must be allowed
            # to correct corrupted observed spins.
            reconstruction = raw_reconstruction

        temperature = self.temperature_head(h5)
        phase_logits = self.phase_head(h5)
        return reconstruction, temperature, phase_logits


# =============================================================================
# Physics-guided objective
# =============================================================================

def neighbor_sum(spins: torch.Tensor) -> torch.Tensor:
    return (
        torch.roll(spins, 1, dims=2)
        + torch.roll(spins, -1, dims=2)
        + torch.roll(spins, 1, dims=3)
        + torch.roll(spins, -1, dims=3)
    )


def energy_per_site(spins: torch.Tensor) -> torch.Tensor:
    return -(
        spins * neighbor_sum(spins)
    ).mean(dim=(1, 2, 3)) / 2.0


def nearest_neighbor_correlation(spins: torch.Tensor) -> torch.Tensor:
    """
    Nearest-neighbor correlation C1.

    For the nearest-neighbor Ising model with J=1, energy and C1 are
    analytically dependent (E = -2*C1). Therefore, C1 is retained only
    as an evaluation observable and is not used as an independent loss.
    """
    return (
        spins * neighbor_sum(spins)
    ).mean(dim=(1, 2, 3)) / 4.0


def second_neighbor_sum(spins: torch.Tensor) -> torch.Tensor:
    """Four axial neighbors at separation two with periodic boundaries."""
    return (
        torch.roll(spins, 2, dims=2)
        + torch.roll(spins, -2, dims=2)
        + torch.roll(spins, 2, dims=3)
        + torch.roll(spins, -2, dims=3)
    )


def second_neighbor_correlation(spins: torch.Tensor) -> torch.Tensor:
    """Second-neighbor correlation C2 at axial separation two."""
    return (
        spins * second_neighbor_sum(spins)
    ).mean(dim=(1, 2, 3)) / 4.0


def absolute_magnetization(spins: torch.Tensor) -> torch.Tensor:
    return spins.mean(dim=(2, 3)).abs().squeeze(1)


def binary_consistency_loss(prediction: torch.Tensor) -> torch.Tensor:
    return ((prediction.abs() - 1.0) ** 2).mean()


def compute_loss(
    model: EnhancedInvNet,
    clean: torch.Tensor,
    temperature_target: torch.Tensor,
    phase_target: torch.Tensor,
    corruption_type: str,
    corruption_level: float,
    phase_criterion: nn.Module,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    model_input, corrupted, affected = apply_corruption(
        clean,
        corruption_type,
        corruption_level,
    )

    preserve_observed = corruption_type == "mask"
    prediction, temperature_prediction, phase_logits = model(
        model_input,
        preserve_observed=preserve_observed,
    )

    if corruption_type == "mask":
        observation_mask = (~affected).float()
        missing_mask = affected.float()

        observed_loss = (
            observation_mask * (prediction - clean).abs()
        ).sum() / (observation_mask.sum() + 1e-8)

        missing_loss = (
            missing_mask * (prediction - clean).abs()
        ).sum() / (missing_mask.sum() + 1e-8)

        reconstruction_loss = observed_loss + 2.0 * missing_loss
    else:
        reconstruction_loss = (prediction - clean).abs().mean()

    energy_loss = (
        energy_per_site(prediction).mean()
        - energy_per_site(clean).mean().detach()
    ).pow(2)

    magnetization_loss = (
        absolute_magnetization(prediction).mean()
        - absolute_magnetization(clean).mean().detach()
    ).pow(2)

    correlation_loss = (
        second_neighbor_correlation(prediction).mean()
        - second_neighbor_correlation(clean).mean().detach()
    ).pow(2)

    bin_loss = binary_consistency_loss(prediction)
    temperature_loss = (temperature_prediction - temperature_target).abs().mean()
    phase_loss = phase_criterion(phase_logits, phase_target)

    total_loss = (
        LAM_REC * reconstruction_loss
        + LAM_E * energy_loss
        + LAM_M * magnetization_loss
        + LAM_C2 * correlation_loss
        + LAM_T * temperature_loss
        + LAM_PHASE * phase_loss
        + LAM_BIN * bin_loss
    )

    components = {
        "total": float(total_loss.detach().item()),
        "reconstruction": float(reconstruction_loss.detach().item()),
        "energy": float(energy_loss.detach().item()),
        "magnetization": float(magnetization_loss.detach().item()),
        "correlation": float(correlation_loss.detach().item()),
        "binary": float(bin_loss.detach().item()),
        "temperature": float(temperature_loss.detach().item()),
        "phase": float(phase_loss.detach().item()),
    }
    return total_loss, components


def average_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        raise RuntimeError("No batches were processed.")
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


@torch.no_grad()
def validate_model(
    model: EnhancedInvNet,
    loader: DataLoader,
    device: torch.device,
    corruption_type: str,
    corruption_level: float,
    seed: int,
    phase_criterion: nn.Module,
) -> Dict[str, float]:
    # The same validation corruption is reused every epoch.
    set_seed(seed)
    model.eval()
    rows: List[Dict[str, float]] = []

    for batch in loader:
        clean = batch["S"].to(device, non_blocking=True).float()
        temperature = batch["T"].to(device, non_blocking=True).float()
        phase = batch["P"].to(device, non_blocking=True).long()

        _, components = compute_loss(
            model,
            clean,
            temperature,
            phase,
            corruption_type,
            corruption_level,
            phase_criterion,
        )
        rows.append(components)

    return average_rows(rows)


def train_one_model(
    model_name: str,
    corruption_type: str,
    training_level: float,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> Tuple[EnhancedInvNet, pd.DataFrame]:
    print("\n" + "=" * 78)
    print(
        f"Training {model_name}: {corruption_type} level "
        f"{100 * training_level:.0f}%"
    )
    print("=" * 78)

    set_seed(SEED)
    model = EnhancedInvNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )
    phase_criterion = nn.CrossEntropyLoss()

    model_dir = OUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best_model.pth"

    best_validation_loss = math.inf
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        train_rows: List[Dict[str, float]] = []

        for batch in train_loader:
            clean = batch["S"].to(device, non_blocking=True).float()
            temperature = batch["T"].to(device, non_blocking=True).float()
            phase = batch["P"].to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            loss, components = compute_loss(
                model,
                clean,
                temperature,
                phase,
                corruption_type,
                training_level,
                phase_criterion,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_rows.append(components)

        train_metrics = average_rows(train_rows)
        validation_metrics = validate_model(
            model,
            validation_loader,
            device,
            corruption_type,
            training_level,
            seed=SEED + 10_000,
            phase_criterion=phase_criterion,
        )
        scheduler.step()

        improved = validation_metrics["total"] < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics["total"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_loss": best_validation_loss,
                    "corruption_type": corruption_type,
                    "training_level": training_level,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        history.append(
            {
                "Model": model_name,
                "Epoch": epoch,
                "Train_Total_Loss": train_metrics["total"],
                "Train_Reconstruction_Loss": train_metrics["reconstruction"],
                "Validation_Total_Loss": validation_metrics["total"],
                "Validation_Reconstruction_Loss": validation_metrics[
                    "reconstruction"
                ],
                "Learning_Rate": scheduler.get_last_lr()[0],
                "Epoch_Seconds": time.time() - epoch_start,
            }
        )

        marker = "*" if improved else " "
        print(
            f"{marker} Epoch {epoch:03d}/{EPOCHS} | "
            f"train={train_metrics['total']:.5f} | "
            f"val={validation_metrics['total']:.5f} | "
            f"val_rec={validation_metrics['reconstruction']:.5f} | "
            f"patience={patience_counter}/{PATIENCE}"
        )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    history_dataframe = pd.DataFrame(history)
    history_dataframe.to_csv(model_dir / "training_history.csv", index=False)
    return model, history_dataframe


@torch.no_grad()
def evaluate_level(
    model: EnhancedInvNet,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    corruption_type: str,
    training_level: float,
    test_level: float,
    evaluation_seed: int,
) -> Dict[str, float]:
    set_seed(evaluation_seed)
    model.eval()

    number_samples = 0
    number_sites = 0
    affected_sites = 0
    unaffected_sites = 0
    correct_all = 0
    correct_affected = 0
    correct_unaffected = 0

    temperature_absolute_error = 0.0
    phase_correct = 0

    energy_absolute_error = 0.0
    magnetization_absolute_error = 0.0
    c1_absolute_error = 0.0
    c2_absolute_error = 0.0

    for batch in loader:
        clean = batch["S"].to(device, non_blocking=True).float()
        temperature = batch["T"].to(device, non_blocking=True).float()
        phase = batch["P"].to(device, non_blocking=True).long()

        model_input, _, affected = apply_corruption(
            clean,
            corruption_type,
            test_level,
        )
        prediction, temperature_prediction, phase_logits = model(
            model_input,
            preserve_observed=(corruption_type == "mask"),
        )

        binary_prediction = torch.where(
            prediction >= 0.0,
            1.0,
            -1.0,
        )
        correct = binary_prediction.eq(clean)

        unaffected = ~affected
        batch_size = clean.shape[0]

        number_samples += batch_size
        number_sites += clean.numel()
        affected_sites += int(affected.sum().item())
        unaffected_sites += int(unaffected.sum().item())

        correct_all += int(correct.sum().item())
        correct_affected += int((correct & affected).sum().item())
        correct_unaffected += int((correct & unaffected).sum().item())

        temperature_absolute_error += float(
            (temperature_prediction - temperature).abs().sum().item()
        )
        phase_correct += int(
            (phase_logits.argmax(dim=1) == phase).sum().item()
        )

        true_energy = energy_per_site(clean)
        predicted_energy = energy_per_site(binary_prediction)
        true_magnetization = absolute_magnetization(clean)
        predicted_magnetization = absolute_magnetization(binary_prediction)
        true_c1 = nearest_neighbor_correlation(clean)
        predicted_c1 = nearest_neighbor_correlation(binary_prediction)
        true_c2 = second_neighbor_correlation(clean)
        predicted_c2 = second_neighbor_correlation(binary_prediction)

        energy_absolute_error += float(
            (predicted_energy - true_energy).abs().sum().item()
        )
        magnetization_absolute_error += float(
            (predicted_magnetization - true_magnetization).abs().sum().item()
        )
        c1_absolute_error += float(
            (predicted_c1 - true_c1).abs().sum().item()
        )
        c2_absolute_error += float(
            (predicted_c2 - true_c2).abs().sum().item()
        )

    if affected_sites == 0:
        raise RuntimeError("No corrupted/masked sites were generated.")

    return {
        "L": L,
        "Model": model_name,
        "Corruption_Type": corruption_type,
        "Training_Level": training_level,
        "Test_Level": test_level,
        "Test_Level_Percent": 100.0 * test_level,
        "Overall_Reconstruction_Accuracy": correct_all / number_sites,
        "Affected_Site_Accuracy": correct_affected / affected_sites,
        "Unaffected_Site_Accuracy": (
            correct_unaffected / max(1, unaffected_sites)
        ),
        "Temperature_MAE": temperature_absolute_error / number_samples,
        "Phase_Accuracy": phase_correct / number_samples,
        "Energy_MAE": energy_absolute_error / number_samples,
        "Magnetization_MAE": magnetization_absolute_error / number_samples,
        "C1_MAE": c1_absolute_error / number_samples,
        "C2_MAE": c2_absolute_error / number_samples,
        "N_Test": number_samples,
        "N_Affected_Sites": affected_sites,
    }


def save_robustness_plots(summary: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        for metric, title, ylabel in (
            (
                "Affected_Site_Accuracy",
                "Accuracy on corrupted or missing sites",
                "Affected-site accuracy",
            ),
            (
                "Overall_Reconstruction_Accuracy",
                "Overall reconstruction robustness",
                "Overall reconstruction accuracy",
            ),
            (
                "Temperature_MAE",
                "Temperature prediction robustness",
                "Temperature MAE",
            ),
            (
                "Phase_Accuracy",
                "Phase classification robustness",
                "Phase accuracy",
            ),
        ):
            figure = plt.figure(figsize=(7.2, 5.0))
            axis = figure.add_subplot(111)

            for corruption_type in ("noise", "mask"):
                subset = summary[
                    summary["Corruption_Type"] == corruption_type
                ].sort_values("Test_Level_Percent")

                axis.plot(
                    subset["Test_Level_Percent"],
                    subset[metric],
                    marker="o",
                    label=corruption_type.capitalize(),
                )

            axis.set_xlabel("Corruption or masking level (%)")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
            axis.legend()
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)


def main() -> None:
    validate_settings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = resolve_device()

    print("=" * 78)
    print("REVIEWER 6: MINIMAL NOISE AND MASKING ROBUSTNESS")
    print("=" * 78)
    print(f"Dataset             : {DATASET_NAME}")
    print(f"Clean CSV           : {CLEAN_CSV}")
    print(f"Reference noisy CSV : {REFERENCE_NOISY_CSV}")
    print(
        "Note: controlled robustness levels are generated directly from "
        "the clean configurations."
    )
    print(f"Output directory    : {OUT_DIR}")
    print(f"Batch size          : {BATCH_SIZE}")

    clean_spins, temperatures, phases = load_clean_data(CLEAN_CSV, L)
    dataset = CleanIsingDataset(clean_spins, temperatures, phases)
    train_loader, validation_loader, test_loader = make_loaders(dataset)

    print(
        f"Split: train={len(train_loader.dataset):,}, "
        f"validation={len(validation_loader.dataset):,}, "
        f"test={len(test_loader.dataset):,}"
    )

    all_results: List[Dict[str, float]] = []
    all_histories: List[pd.DataFrame] = []

    # Model 1: trained once at moderate spin-flip noise.
    noise_model, noise_history = train_one_model(
        model_name="Noise_Model_Train30",
        corruption_type="noise",
        training_level=NOISE_TRAIN_LEVEL,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
    )
    all_histories.append(noise_history)

    for level_index, test_level in enumerate(NOISE_TEST_LEVELS):
        result = evaluate_level(
            noise_model,
            test_loader,
            device,
            model_name="Noise_Model_Train30",
            corruption_type="noise",
            training_level=NOISE_TRAIN_LEVEL,
            test_level=test_level,
            evaluation_seed=SEED + 20_000 + level_index,
        )
        all_results.append(result)
        print(
            f"Noise test {100 * test_level:.0f}%: "
            f"affected accuracy={result['Affected_Site_Accuracy']:.4f}, "
            f"overall accuracy={result['Overall_Reconstruction_Accuracy']:.4f}"
        )

    # Free GPU memory before training the masking model.
    del noise_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Model 2: trained once at moderate random masking.
    mask_model, mask_history = train_one_model(
        model_name="Mask_Model_Train40",
        corruption_type="mask",
        training_level=MASK_TRAIN_LEVEL,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
    )
    all_histories.append(mask_history)

    for level_index, test_level in enumerate(MASK_TEST_LEVELS):
        result = evaluate_level(
            mask_model,
            test_loader,
            device,
            model_name="Mask_Model_Train40",
            corruption_type="mask",
            training_level=MASK_TRAIN_LEVEL,
            test_level=test_level,
            evaluation_seed=SEED + 30_000 + level_index,
        )
        all_results.append(result)
        print(
            f"Mask test {100 * test_level:.0f}%: "
            f"imputation accuracy={result['Affected_Site_Accuracy']:.4f}, "
            f"overall accuracy={result['Overall_Reconstruction_Accuracy']:.4f}"
        )

    summary = pd.DataFrame(all_results)
    summary_path = OUT_DIR / f"robustness_summary_L{L}.csv"
    history_path = OUT_DIR / f"robustness_training_history_L{L}.csv"
    plots_path = OUT_DIR / f"robustness_curves_L{L}.pdf"
    protocol_path = OUT_DIR / f"robustness_protocol_L{L}.txt"

    summary.to_csv(summary_path, index=False)
    pd.concat(all_histories, ignore_index=True).to_csv(
        history_path,
        index=False,
    )
    save_robustness_plots(summary, plots_path)

    with protocol_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Dataset={DATASET_NAME}\n")
        handle.write(f"Clean_CSV={CLEAN_CSV}\n")
        handle.write(
            "Noise_protocol=train at 30% spin-flip noise; "
            "test at 10%, 30%, and 50%\n"
        )
        handle.write(
            "Mask_protocol=train at 40% random masking; "
            "test at 20%, 40%, and 60%\n"
        )
        handle.write(
            "Controlled corruptions were generated directly from "
            "clean Monte Carlo configurations.\n"
        )
        handle.write(f"Seed={SEED}\n")
        handle.write(f"LAM_E={LAM_E}\n")
        handle.write(f"LAM_M={LAM_M}\n")
        handle.write(f"LAM_C2={LAM_C2}\n")
        handle.write(f"LAM_BIN={LAM_BIN}\n")
        handle.write("C1=evaluation only; not used as an independent loss because E=-2*C1 for J=1.\n")
        handle.write("C2=independent correlation regularizer at axial separation two with periodic boundaries.\n")

    print("\n" + "=" * 78)
    print("ROBUSTNESS EVALUATION COMPLETED")
    print("=" * 78)
    display_columns = [
        "Corruption_Type",
        "Training_Level",
        "Test_Level",
        "Affected_Site_Accuracy",
        "Overall_Reconstruction_Accuracy",
        "Temperature_MAE",
        "Phase_Accuracy",
    ]
    print(summary[display_columns].to_string(index=False))
    print(f"\nSummary CSV : {summary_path}")
    print(f"History CSV : {history_path}")
    print(f"Curves PDF  : {plots_path}")
    print(f"Protocol    : {protocol_path}")


if __name__ == "__main__":
    main()
