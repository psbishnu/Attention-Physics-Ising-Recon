#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Ablation7.py

This program:
1. Loads the existing EnhancedInvNet checkpoint produced by the main J5DL code.
2. Reconstructs the same deterministic 10% test split.
3. Extracts the model's learned spatial attention maps.
4. Quantifies whether attention focuses on:
   - domain boundaries,
   - locally disagreeing/fluctuating spins,
   - the near-critical temperature regime.
5. Uses an unbiased shuffled-attention AUROC (~0.5) as the random control.

NO model retraining is performed.
NO Restormer is used.

Main outputs:
    attention_per_sample_L{L}.csv
    attention_summary_regimes_L{L}.csv
    attention_summary_temperature_L{L}.csv
    attention_physics_curves_L{L}.pdf
    attention_examples_L{L}.pdf
    attention_table_L{L}.tex

Run on HPC:
    python Ablation7.py

For another lattice size, change only DATASET_NAME and, when necessary,
CHECKPOINT_OVERRIDE. 
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------------------------------------------------------
# PDF typography only: larger and bold fonts.
# Numerical analysis, model loading, metrics, data handling, and outputs are
# unchanged.
# -------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 15,
    "font.weight": "bold",
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "axes.labelweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
    "figure.titleweight": "bold",
})



# =============================================================================
# USER SETTINGS: EDIT ONLY THIS SECTION
# =============================================================================

L = 32  # Change only this value: 32, 64, or 128
DATASET_NAME = f"MCD{L}"

# Ablation7.py is assumed to be inside the folder "scripta".
# Data are assumed to be inside ../JOB5_Noise/J5Data/.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR / "../JOB5_Noise/J5Data").resolve()

# L is defined above and DATASET_NAME is derived from it.
CLEAN_CSV = DATA_DIR / f"MCD{L}.csv"
NOISY_CSV = DATA_DIR / f"MCDN{L}.csv"

# Explicit HPC checkpoint path. Change only L above for L=32, 64, or 128.
CHECKPOINT_OVERRIDE = f"/home/partha02965/JOB_SCRIPTA/Generated_L{L}/best_model.pth"

OUT_DIR = SCRIPT_DIR / f"Ablation7_Final_Attention_{DATASET_NAME}"

DEVICE = "cuda"                     # Automatically falls back to CPU
BATCH_SIZE = {32: 64, 64: 16, 128: 2}[L]
NUM_WORKERS = 0
SEED = 123

TC = 2.269
CRITICAL_WIDTH = 0.20
N_SHUFFLES = 10

# Test rows are loaded in chunks to reduce memory usage, especially for L=128.
CSV_CHUNK_SIZE = {32: 4096, 64: 1024, 128: 128}[L]

# =============================================================================


# =============================================================================
# General utilities
# =============================================================================

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
        print("CUDA requested but unavailable; using CPU.")
    else:
        print("Using CPU.")
    return torch.device("cpu")


def resolve_checkpoint() -> Path:
    if CHECKPOINT_OVERRIDE.strip():
        path = Path(CHECKPOINT_OVERRIDE)
        if not path.is_absolute():
            path = (SCRIPT_DIR / path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"CHECKPOINT_OVERRIDE does not exist:\n{path}"
            )
        return path

    candidates = [
        (SCRIPT_DIR / f"../JOB5_Noise/Generated_L{L}/best_model.pth").resolve(),
        (SCRIPT_DIR / f"../JOB5_Noise/Generated_{DATASET_NAME}/best_model.pth").resolve(),
        (SCRIPT_DIR / f"Generated_L{L}/best_model.pth").resolve(),
        (SCRIPT_DIR / f"Generated_{DATASET_NAME}/best_model.pth").resolve(),
    ]

    existing = []
    for path in candidates:
        if path.exists() and path not in existing:
            existing.append(path)

    if len(existing) == 1:
        return existing[0]

    if len(existing) > 1:
        paths = "\n".join(f"  - {path}" for path in existing)
        raise RuntimeError(
            "Multiple checkpoints were found. Set CHECKPOINT_OVERRIDE to "
            f"the correct one:\n{paths}"
        )

    search_root = (SCRIPT_DIR / "../JOB5_Noise").resolve()
    if search_root.exists():
        matching = [
            path.resolve()
            for path in search_root.rglob("best_model.pth")
            if (
                f"L{L}" in str(path.parent)
                or DATASET_NAME in str(path.parent)
            )
        ]
        matching = list(dict.fromkeys(matching))

        if len(matching) == 1:
            return matching[0]

        if len(matching) > 1:
            paths = "\n".join(f"  - {path}" for path in matching)
            raise RuntimeError(
                "Multiple possible checkpoints were found. Set "
                f"CHECKPOINT_OVERRIDE explicitly:\n{paths}"
            )

    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "The trained EnhancedInvNet checkpoint was not found.\n"
        "Set CHECKPOINT_OVERRIDE to the exact best_model.pth path.\n"
        f"Attempted paths:\n{attempted}"
    )


def validate_inputs(checkpoint_path: Path) -> None:
    if L not in (32, 64, 128):
        raise ValueError("DATASET_NAME must be MCD32, MCD64, or MCD128.")
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"Clean CSV not found:\n{CLEAN_CSV}")
    if not NOISY_CSV.exists():
        raise FileNotFoundError(f"Noisy CSV not found:\n{NOISY_CSV}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")


def extract_spin_columns(
    dataframe_columns: Sequence[str],
    lattice_size: int,
) -> List[str]:
    spin_columns = [
        column
        for column in dataframe_columns
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
        unique_values = set(np.unique(values).tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Numeric phase labels must be 0/1; found {unique_values}."
            )
        return values

    normalized = series.astype(str).str.strip().str.upper()
    mapped = normalized.map({"F": 1, "P": 0})

    if mapped.isna().any():
        unexpected = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"Unexpected phase labels: {unexpected}")

    return mapped.to_numpy(dtype=np.int64)


# =============================================================================
# Reproduce the main code's deterministic 80/10/10 test split
# =============================================================================

def number_of_rows(csv_path: Path) -> int:
    temperature_only = pd.read_csv(csv_path, usecols=["Temperature"])
    return int(len(temperature_only))


def deterministic_test_indices(
    number_samples: int,
    seed: int,
) -> List[int]:
    number_train = int(0.8 * number_samples)
    number_validation = int(0.1 * number_samples)
    number_test = number_samples - number_train - number_validation

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(
        number_samples,
        generator=generator,
    ).tolist()

    return permutation[
        number_train + number_validation:
        number_train + number_validation + number_test
    ]


def read_selected_rows(
    csv_path: Path,
    selected_indices: Sequence[int],
    usecols: Sequence[str],
    dtype_map: Dict[str, object],
    chunk_size: int,
) -> pd.DataFrame:
    selected_set = set(int(index) for index in selected_indices)
    pieces: List[pd.DataFrame] = []
    offset = 0

    for chunk in pd.read_csv(
        csv_path,
        usecols=list(usecols),
        dtype=dtype_map,
        chunksize=chunk_size,
    ):
        chunk_start = offset
        chunk_end = offset + len(chunk)

        local_positions = [
            global_index - chunk_start
            for global_index in selected_set
            if chunk_start <= global_index < chunk_end
        ]

        if local_positions:
            selected_chunk = chunk.iloc[sorted(local_positions)].copy()
            selected_chunk["_GlobalIndex"] = [
                chunk_start + local_position
                for local_position in sorted(local_positions)
            ]
            pieces.append(selected_chunk)

        offset = chunk_end

    if not pieces:
        raise RuntimeError(f"No selected rows were loaded from {csv_path}.")

    dataframe = pd.concat(pieces, ignore_index=True)
    if len(dataframe) != len(selected_indices):
        raise RuntimeError(
            f"Expected {len(selected_indices)} selected rows from {csv_path}, "
            f"but loaded {len(dataframe)}."
        )

    # Restore exactly the same sample order as the original random_split test subset.
    order_map = {
        int(global_index): position
        for position, global_index in enumerate(selected_indices)
    }
    dataframe["_Order"] = dataframe["_GlobalIndex"].map(order_map)
    dataframe = (
        dataframe
        .sort_values("_Order")
        .drop(columns="_Order")
        .reset_index(drop=True)
    )
    return dataframe


def load_test_data(
    clean_path: Path,
    noisy_path: Path,
    lattice_size: int,
    seed: int,
) -> Tuple[np.ndarray, ...]:
    clean_header = pd.read_csv(clean_path, nrows=0)
    noisy_header = pd.read_csv(noisy_path, nrows=0)

    spin_columns = extract_spin_columns(
        clean_header.columns.tolist(),
        lattice_size,
    )

    missing_noisy_columns = [
        column for column in spin_columns
        if column not in noisy_header.columns
    ]
    if missing_noisy_columns:
        raise ValueError(
            f"The noisy CSV is missing {len(missing_noisy_columns)} "
            "spin columns."
        )

    total_samples = number_of_rows(clean_path)
    test_indices = deterministic_test_indices(total_samples, seed)

    print(
        f"Total samples={total_samples:,}; "
        f"loading only test samples={len(test_indices):,}."
    )

    clean_usecols = ["Temperature", "Phase", *spin_columns]
    noisy_usecols = spin_columns

    clean_dtype = {column: np.int8 for column in spin_columns}
    clean_dtype["Temperature"] = np.float32
    noisy_dtype = {column: np.int8 for column in spin_columns}

    clean_df = read_selected_rows(
        clean_path,
        test_indices,
        clean_usecols,
        clean_dtype,
        CSV_CHUNK_SIZE,
    )
    noisy_df = read_selected_rows(
        noisy_path,
        test_indices,
        noisy_usecols,
        noisy_dtype,
        CSV_CHUNK_SIZE,
    )

    clean_indices = clean_df["_GlobalIndex"].to_numpy(dtype=np.int64)
    noisy_indices = noisy_df["_GlobalIndex"].to_numpy(dtype=np.int64)

    if not np.array_equal(clean_indices, noisy_indices):
        raise RuntimeError("Clean and noisy test rows are not aligned.")

    clean_spins = (
        clean_df[spin_columns]
        .to_numpy(dtype=np.float32)
        .reshape(-1, lattice_size, lattice_size)
    )
    noisy_spins = (
        noisy_df[spin_columns]
        .to_numpy(dtype=np.float32)
        .reshape(-1, lattice_size, lattice_size)
    )
    temperatures = clean_df["Temperature"].to_numpy(dtype=np.float32)
    phases = phase_to_int(clean_df["Phase"])
    masks = (noisy_spins != 0).astype(np.float32)

    model_inputs = np.stack(
        (noisy_spins, masks),
        axis=1,
    ).astype(np.float32)

    return (
        model_inputs,
        clean_spins,
        temperatures,
        phases,
        clean_indices,
    )


class TestIsingDataset(Dataset):
    def __init__(
        self,
        model_inputs: np.ndarray,
        clean_spins: np.ndarray,
        temperatures: np.ndarray,
        phases: np.ndarray,
        dataset_indices: np.ndarray,
    ) -> None:
        self.model_inputs = torch.from_numpy(model_inputs)
        self.clean_spins = torch.from_numpy(clean_spins)[:, None]
        self.temperatures = torch.from_numpy(temperatures)[:, None]
        self.phases = torch.from_numpy(phases)
        self.dataset_indices = torch.from_numpy(dataset_indices)

    def __len__(self) -> int:
        return self.model_inputs.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "X": self.model_inputs[index],
            "S": self.clean_spins[index],
            "T": self.temperatures[index],
            "P": self.phases[index],
            "DatasetIndex": self.dataset_indices[index],
        }


# =============================================================================
# Exact EnhancedInvNet architecture from the supplied main code
# =============================================================================

class EnhancedInvNet(nn.Module):
    def __init__(self, lattice_size: int) -> None:
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

        self.head_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.head_P = nn.Sequential(
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
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        noisy_input = model_input[:, :1]
        mask = model_input[:, 1:2]

        h1 = self.activation(self.enc_conv1(model_input))
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
        raw_reconstruction = torch.tanh(self.dec_conv5(d4))

        reconstruction = (
            mask * noisy_input
            + (1.0 - mask) * raw_reconstruction
        )

        temperature_prediction = self.head_T(h5)
        phase_logits = self.head_P(h5)

        return (
            reconstruction,
            temperature_prediction,
            phase_logits,
            attention_map,
        )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> EnhancedInvNet:
    model = EnhancedInvNet(L).to(device)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    epoch = (
        checkpoint.get("epoch", "unknown")
        if isinstance(checkpoint, dict)
        else "unknown"
    )
    print(f"Loaded trained model from epoch: {epoch}")
    return model


# =============================================================================
# Physical structures and quantitative metrics
# =============================================================================

def local_disagreement_map(spins: np.ndarray) -> np.ndarray:
    up = np.roll(spins, 1, axis=0)
    down = np.roll(spins, -1, axis=0)
    left = np.roll(spins, 1, axis=1)
    right = np.roll(spins, -1, axis=1)

    return (
        (spins != up).astype(np.float64)
        + (spins != down).astype(np.float64)
        + (spins != left).astype(np.float64)
        + (spins != right).astype(np.float64)
    ) / 4.0


def domain_boundary_map(spins: np.ndarray) -> np.ndarray:
    return local_disagreement_map(spins) > 0.0


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)

    start = 0
    while start < values.size:
        end = start + 1
        while (
            end < values.size
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def binary_roc_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    positives = labels == 1
    negatives = labels == 0
    number_positive = int(positives.sum())
    number_negative = int(negatives.sum())

    if number_positive == 0 or number_negative == 0:
        return float("nan")

    ranks = average_ranks(scores)
    positive_rank_sum = ranks[positives].sum()

    auc = (
        positive_rank_sum
        - number_positive * (number_positive + 1) / 2.0
    ) / (number_positive * number_negative)

    return float(auc)


def safe_pearson(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)

    if (
        first.size < 2
        or np.std(first) < 1e-12
        or np.std(second) < 1e-12
    ):
        return float("nan")

    return float(np.corrcoef(first, second)[0, 1])


def temperature_regime(temperature: float) -> str:
    if temperature < TC - CRITICAL_WIDTH:
        return "Ordered"
    if temperature > TC + CRITICAL_WIDTH:
        return "Disordered"
    return "Near-critical"


def analyze_one_sample(
    attention: np.ndarray,
    clean_spins: np.ndarray,
    random_generator: np.random.Generator,
) -> Dict[str, float]:
    """
    Quantify both the signed/raw attention response and its orientation-independent
    physical separability.

    Important:
    In the main EnhancedInvNet, the sigmoid attention map is concatenated with
    encoder features rather than multiplied as a conventional positive gate.
    Therefore high numerical values are not guaranteed to mean "more important".
    A physically meaningful boundary can be encoded either by enhancement
    (A_boundary > A_interior) or by suppression (A_boundary < A_interior).

    We therefore retain the raw metrics AND report:
      - Boundary_Contrast_Factor = max(R_A, 1/R_A)
      - Boundary_Separability_AUROC = max(AUC, 1-AUC) for learned attention
      - Attention_Disagreement_Strength = |rho|

    The sign/direction is NOT discarded: raw R_A, raw AUROC and raw rho are
    retained in every CSV, and group summaries explicitly label the response as
    "Boundary-enhanced" or "Boundary-suppressed".
    """
    disagreement = local_disagreement_map(clean_spins)
    boundary = disagreement > 0.0
    interior = ~boundary

    mean_boundary_attention = (
        float(attention[boundary].mean())
        if boundary.any()
        else float("nan")
    )
    mean_interior_attention = (
        float(attention[interior].mean())
        if interior.any()
        else float("nan")
    )

    enrichment_ratio = (
        mean_boundary_attention / (mean_interior_attention + 1e-12)
        if (
            np.isfinite(mean_boundary_attention)
            and np.isfinite(mean_interior_attention)
        )
        else float("nan")
    )

    suppression_ratio = (
        mean_interior_attention / (mean_boundary_attention + 1e-12)
        if (
            np.isfinite(mean_boundary_attention)
            and np.isfinite(mean_interior_attention)
        )
        else float("nan")
    )

    if np.isfinite(enrichment_ratio) and enrichment_ratio > 0.0:
        contrast_factor = max(enrichment_ratio, 1.0 / enrichment_ratio)
    else:
        contrast_factor = float("nan")

    boundary_labels = boundary.astype(np.int64).reshape(-1)
    attention_scores = attention.reshape(-1)

    raw_boundary_auc = binary_roc_auc(
        boundary_labels,
        attention_scores,
    )

    boundary_separability_auc = (
        max(raw_boundary_auc, 1.0 - raw_boundary_auc)
        if np.isfinite(raw_boundary_auc)
        else float("nan")
    )

    # Random-control AUROC:
    # Keep the shuffled AUROC in its original orientation. Transforming each
    # random AUROC with max(AUC, 1-AUC) would bias the random baseline upward
    # above 0.5. A valid shuffled control should remain approximately 0.5.
    shuffled_raw_aucs = []
    for _ in range(max(1, N_SHUFFLES)):
        shuffled_attention = random_generator.permutation(attention_scores)
        shuffled_auc = binary_roc_auc(
            boundary_labels,
            shuffled_attention,
        )
        shuffled_raw_aucs.append(shuffled_auc)

    correlation = safe_pearson(
        attention_scores,
        disagreement.reshape(-1),
    )
    correlation_strength = (
        abs(correlation)
        if np.isfinite(correlation)
        else float("nan")
    )

    return {
        # Raw/signed metrics: preserve these for complete transparency.
        "MeanAttention_Boundary": mean_boundary_attention,
        "MeanAttention_Interior": mean_interior_attention,
        "BoundaryEnrichment_Ratio": enrichment_ratio,
        "Boundary_AUROC": raw_boundary_auc,
        "Shuffled_AUROC": float(np.nanmean(shuffled_raw_aucs)),
        "Attention_Disagreement_Correlation": correlation,

        # Orientation-aware metrics: quantify whether the map distinguishes
        # physical structure even when the learned polarity is suppressive.
        "Boundary_Suppression_Ratio": suppression_ratio,
        "Boundary_Contrast_Factor": contrast_factor,
        "Boundary_Separability_AUROC": boundary_separability_auc,
        # Use the untransformed shuffled AUROC as the random baseline.
        "Shuffled_Separability_AUROC": float(
            np.nanmean(shuffled_raw_aucs)
        ),
        "Attention_Disagreement_Strength": correlation_strength,

        "BoundaryFraction": float(boundary.mean()),
        "MeanLocalDisagreement": float(disagreement.mean()),
    }


METRIC_COLUMNS = [
    "MeanAttention_Boundary",
    "MeanAttention_Interior",
    "BoundaryEnrichment_Ratio",
    "Boundary_Suppression_Ratio",
    "Boundary_Contrast_Factor",
    "Boundary_AUROC",
    "Boundary_Separability_AUROC",
    "Shuffled_AUROC",
    "Shuffled_Separability_AUROC",
    "Attention_Disagreement_Correlation",
    "Attention_Disagreement_Strength",
    "BoundaryFraction",
    "MeanLocalDisagreement",
]


def summarize_group(
    dataframe: pd.DataFrame,
    group_label: str,
) -> Dict[str, float]:
    row: Dict[str, float] = {
        "L": L,
        "Group": group_label,
        "N_samples": int(len(dataframe)),
        "Mean_Temperature": float(dataframe["Temperature"].mean()),
    }

    for column in METRIC_COLUMNS:
        values = dataframe[column].to_numpy(dtype=np.float64)
        finite_values = values[np.isfinite(values)]

        row[f"{column}_Mean"] = (
            float(np.mean(finite_values))
            if finite_values.size
            else float("nan")
        )
        row[f"{column}_Std"] = (
            float(np.std(finite_values, ddof=1))
            if finite_values.size > 1
            else 0.0
        )
        row[f"{column}_SEM"] = (
            float(np.std(finite_values, ddof=1) / math.sqrt(finite_values.size))
            if finite_values.size > 1
            else 0.0
        )

    mean_boundary = row["MeanAttention_Boundary_Mean"]
    mean_interior = row["MeanAttention_Interior_Mean"]
    mean_rho = row["Attention_Disagreement_Correlation_Mean"]

    if np.isfinite(mean_boundary) and np.isfinite(mean_interior):
        row["Boundary_Response_Direction"] = (
            "Boundary-enhanced"
            if mean_boundary >= mean_interior
            else "Boundary-suppressed"
        )
    else:
        row["Boundary_Response_Direction"] = "Undefined"

    if np.isfinite(mean_rho):
        row["Fluctuation_Response_Direction"] = (
            "Positive"
            if mean_rho >= 0.0
            else "Negative"
        )
    else:
        row["Fluctuation_Response_Direction"] = "Undefined"

    return row


def build_regime_summary(
    per_sample: pd.DataFrame,
) -> pd.DataFrame:
    rows = [summarize_group(per_sample, "Overall")]

    for regime in ("Ordered", "Near-critical", "Disordered"):
        subset = per_sample[per_sample["Regime"] == regime]
        if len(subset):
            rows.append(summarize_group(subset, regime))

    return pd.DataFrame(rows)


def build_temperature_summary(
    per_sample: pd.DataFrame,
) -> pd.DataFrame:
    working = per_sample.copy()
    working["Temperature_Key"] = working["Temperature"].round(6)

    rows = []
    for temperature, subset in working.groupby(
        "Temperature_Key",
        sort=True,
    ):
        row = summarize_group(
            subset,
            f"T={float(temperature):.6f}",
        )
        row["Temperature"] = float(temperature)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Temperature").reset_index(drop=True)


# =============================================================================
# PDF typography helper
# =============================================================================
def apply_bold_large_axis_text(axis):
    axis.tick_params(axis="both", labelsize=13)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight("bold")
    axis.xaxis.label.set_fontweight("bold")
    axis.yaxis.label.set_fontweight("bold")
    axis.title.set_fontweight("bold")


# =============================================================================
# Evaluation and output
# =============================================================================

@torch.no_grad()
def run_attention_analysis(
    model: EnhancedInvNet,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    model.eval()
    random_generator = np.random.default_rng(SEED)

    rows: List[Dict[str, float]] = []
    examples: List[Dict[str, object]] = []

    for batch_number, batch in enumerate(loader, start=1):
        model_input = batch["X"].to(device, non_blocking=True)
        clean_batch = batch["S"][:, 0].numpy()
        temperature_batch = batch["T"][:, 0].numpy()
        phase_batch = batch["P"].numpy()
        dataset_index_batch = batch["DatasetIndex"].numpy()

        (
            _reconstruction,
            _temperature_prediction,
            _phase_logits,
            attention_map,
        ) = model(model_input)

        attention_batch = (
            attention_map[:, 0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        for index in range(len(clean_batch)):
            clean_spins = clean_batch[index]
            attention = attention_batch[index]
            temperature = float(temperature_batch[index])
            regime = temperature_regime(temperature)

            metrics = analyze_one_sample(
                attention,
                clean_spins,
                random_generator,
            )

            row = {
                "L": L,
                "DatasetIndex": int(dataset_index_batch[index]),
                "Temperature": temperature,
                "Phase": int(phase_batch[index]),
                "Regime": regime,
            }
            row.update(metrics)
            rows.append(row)

            examples.append(
                {
                    "L": L,
                    "Temperature": temperature,
                    "Regime": regime,
                    "Clean": clean_spins.copy(),
                    "Boundary": domain_boundary_map(
                        clean_spins
                    ).astype(np.float32),
                    "Disagreement": local_disagreement_map(clean_spins),
                    "Attention": attention.copy(),
                    "Enrichment": metrics["BoundaryEnrichment_Ratio"],
                    "AUROC": metrics["Boundary_AUROC"],
                    "Correlation": metrics[
                        "Attention_Disagreement_Correlation"
                    ],
                    "Contrast": metrics["Boundary_Contrast_Factor"],
                    "SeparabilityAUROC": metrics[
                        "Boundary_Separability_AUROC"
                    ],
                    "AssociationStrength": metrics[
                        "Attention_Disagreement_Strength"
                    ],
                }
            )

        if batch_number % 20 == 0:
            print(
                f"Processed {min(batch_number * BATCH_SIZE, len(loader.dataset)):,}"
                f"/{len(loader.dataset):,} test samples."
            )

    if not rows:
        raise RuntimeError("No test samples were analyzed.")

    per_sample = (
        pd.DataFrame(rows)
        .sort_values(["Temperature", "DatasetIndex"])
        .reset_index(drop=True)
    )
    return per_sample, examples


def save_attention_curves(
    temperature_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Main reviewer-facing plot.

    These three quantities are orientation-aware:
      C_A >= 1: strength of boundary/interior contrast, regardless of whether
                the raw attention is enhanced or suppressed at boundaries.
      AUC_sep >= 0.5: boundary separability independent of score polarity.
      |rho| >= 0: strength of association with local spin disagreement.

    The raw signed diagnostics are saved separately by
    save_raw_attention_diagnostics().
    """
    with PdfPages(output_path) as pdf:
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(7.5, 10.5),
            sharex=True,
        )

        axes[0].errorbar(
            temperature_summary["Temperature"],
            temperature_summary["Boundary_Contrast_Factor_Mean"],
            yerr=temperature_summary["Boundary_Contrast_Factor_SEM"],
            marker="o",
            markersize=3,
            capsize=2,
        )
        axes[0].axhline(1.0, linestyle="--")
        axes[0].axvline(TC, linestyle=":")
        axes[0].set_ylabel(r"$C_A$")
        axes[0].set_title("Boundary/interior attention contrast")

        axes[1].errorbar(
            temperature_summary["Temperature"],
            temperature_summary["Boundary_Separability_AUROC_Mean"],
            yerr=temperature_summary["Boundary_Separability_AUROC_SEM"],
            marker="o",
            markersize=3,
            capsize=2,
            label="Learned attention",
        )
        axes[1].plot(
            temperature_summary["Temperature"],
            temperature_summary["Shuffled_Separability_AUROC_Mean"],
            marker="s",
            markersize=3,
            label="Shuffled control",
        )
        axes[1].axhline(0.5, linestyle="--")
        axes[1].axvline(TC, linestyle=":")
        axes[1].set_ylabel(r"Boundary $AUROC_{\rm sep}$")
        axes[1].set_title("Orientation-independent boundary separability")
        axes[1].legend(prop={'weight': 'bold', 'size': 13})

        axes[2].errorbar(
            temperature_summary["Temperature"],
            temperature_summary["Attention_Disagreement_Strength_Mean"],
            yerr=temperature_summary["Attention_Disagreement_Strength_SEM"],
            marker="o",
            markersize=3,
            capsize=2,
        )
        axes[2].axhline(0.0, linestyle="--")
        axes[2].axvline(TC, linestyle=":")
        axes[2].set_xlabel("Temperature")
        axes[2].set_ylabel(r"$|\rho_{A,D}|$")
        axes[2].set_title("Attention--local-disagreement association strength")

        figure.suptitle(
            f"Orientation-aware attention--physics analysis, L={L}",
            fontweight="bold",
            fontsize=18,
        )
        for ax in axes:
            apply_bold_large_axis_text(ax)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def save_raw_attention_diagnostics(
    temperature_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the original signed/raw metrics for transparency."""
    with PdfPages(output_path) as pdf:
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(7.5, 10.5),
            sharex=True,
        )

        axes[0].errorbar(
            temperature_summary["Temperature"],
            temperature_summary["BoundaryEnrichment_Ratio_Mean"],
            yerr=temperature_summary["BoundaryEnrichment_Ratio_SEM"],
            marker="o",
            markersize=3,
            capsize=2,
        )
        axes[0].axhline(1.0, linestyle="--")
        axes[0].axvline(TC, linestyle=":")
        axes[0].set_ylabel(r"$R_A$")
        axes[0].set_title("Raw boundary-attention enrichment")

        axes[1].errorbar(
            temperature_summary["Temperature"],
            temperature_summary["Boundary_AUROC_Mean"],
            yerr=temperature_summary["Boundary_AUROC_SEM"],
            marker="o",
            markersize=3,
            capsize=2,
            label="Raw attention score",
        )
        axes[1].plot(
            temperature_summary["Temperature"],
            temperature_summary["Shuffled_AUROC_Mean"],
            marker="s",
            markersize=3,
            label="Shuffled control",
        )
        axes[1].axhline(0.5, linestyle="--")
        axes[1].axvline(TC, linestyle=":")
        axes[1].set_ylabel("Raw boundary AUROC")
        axes[1].set_title("Signed boundary response")
        axes[1].legend(prop={'weight': 'bold', 'size': 13})

        axes[2].errorbar(
            temperature_summary["Temperature"],
            temperature_summary[
                "Attention_Disagreement_Correlation_Mean"
            ],
            yerr=temperature_summary[
                "Attention_Disagreement_Correlation_SEM"
            ],
            marker="o",
            markersize=3,
            capsize=2,
        )
        axes[2].axhline(0.0, linestyle="--")
        axes[2].axvline(TC, linestyle=":")
        axes[2].set_xlabel("Temperature")
        axes[2].set_ylabel(r"$\rho_{A,D}$")
        axes[2].set_title("Signed attention--disagreement correlation")

        figure.suptitle(
            f"Raw attention diagnostics, L={L}",
            fontweight="bold",
            fontsize=18,
        )
        for ax in axes:
            apply_bold_large_axis_text(ax)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def select_examples(
    examples: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    targets = [1.0, 2.0, TC, 4.0]
    selected: List[Dict[str, object]] = []
    used_indices = set()

    for target in targets:
        available = [
            (index, example)
            for index, example in enumerate(examples)
            if index not in used_indices
        ]
        if not available:
            break

        selected_index, selected_example = min(
            available,
            key=lambda item: abs(
                float(item[1]["Temperature"]) - target
            ),
        )
        used_indices.add(selected_index)
        selected.append(selected_example)

    return selected


def save_attention_examples(
    examples: List[Dict[str, object]],
    output_path: Path,
) -> None:
    selected = select_examples(examples)

    with PdfPages(output_path) as pdf:
        for example in selected:
            figure, axes = plt.subplots(
                1,
                5,
                figsize=(15.0, 3.8),
            )

            axes[0].imshow(
                example["Clean"],
                cmap="gray",
                vmin=-1,
                vmax=1,
            )
            axes[0].set_title("Clean spins", fontweight="bold", fontsize=15)
            axes[0].axis("off")

            axes[1].imshow(
                example["Boundary"],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[1].set_title("Domain boundary", fontweight="bold", fontsize=15)
            axes[1].axis("off")

            raw_image = axes[2].imshow(
                example["Attention"],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[2].set_title("Raw attention $A$", fontweight="bold", fontsize=15)
            axes[2].axis("off")
            figure.colorbar(
                raw_image,
                ax=axes[2],
                fraction=0.046,
                pad=0.04,
            )

            # This is shown explicitly as a derived suppression score, not as
            # a replacement for the raw attention map.
            suppression_score = 1.0 - np.asarray(example["Attention"])
            suppression_image = axes[3].imshow(
                suppression_score,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[3].set_title("Suppression score $1-A$", fontweight="bold", fontsize=15)
            axes[3].axis("off")
            figure.colorbar(
                suppression_image,
                ax=axes[3],
                fraction=0.046,
                pad=0.04,
            )

            disagreement_image = axes[4].imshow(
                example["Disagreement"],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[4].set_title("Local disagreement", fontweight="bold", fontsize=15)
            axes[4].axis("off")
            figure.colorbar(
                disagreement_image,
                ax=axes[4],
                fraction=0.046,
                pad=0.04,
            )

            figure.suptitle(
                f"L={L}, T={float(example['Temperature']):.3f}, "
                f"{example['Regime']}; "
                f"C_A={float(example['Contrast']):.3f}, "
                f"AUC_sep={float(example['SeparabilityAUROC']):.3f}, "
                f"|rho|={float(example['AssociationStrength']):.3f}",
                fontweight="bold",
                fontsize=18,
            )
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def save_latex_table(
    regime_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Compact manuscript table.

    C_A is an orientation-independent contrast factor:
        C_A = max(R_A, 1/R_A).
    AUROC_sep = max(AUROC, 1-AUROC).
    The Response column preserves the polarity, so a strong inverse response
    is not mislabeled as positive attention.
    """
    rows = regime_summary[
        regime_summary["Group"].isin(
            ["Ordered", "Near-critical", "Disordered"]
        )
    ]

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table*}[t]\n")
        handle.write("\\centering\n")
        handle.write(
            "\\caption{Quantitative physical interpretation of the learned "
            "spatial-attention response. $C_A$ denotes the magnitude of the "
            "boundary--interior contrast, $\\mathrm{AUROC}_{\\rm sep}$ denotes "
            "orientation-independent boundary separability, and "
            "$|\\rho_{A,D}|$ denotes the strength of association with local "
            "spin disagreement. The response direction indicates whether "
            "raw attention is enhanced or suppressed at domain boundaries.}\n"
        )
        handle.write("\\label{tab:attention_physics}\n")
        handle.write("\\begin{tabular}{cclccc}\n")
        handle.write("\\hline\n")
        handle.write(
            "$L$ & Regime & Response & $C_A$ & "
            "$\\mathrm{AUROC}_{\\rm sep}$ & $|\\rho_{A,D}|$ \\\\\n"
        )
        handle.write("\\hline\n")

        for _, row in rows.iterrows():
            handle.write(
                f"{L} & {row['Group']} & "
                f"{row['Boundary_Response_Direction']} & "
                f"{row['Boundary_Contrast_Factor_Mean']:.3f} $\\pm$ "
                f"{row['Boundary_Contrast_Factor_SEM']:.3f} & "
                f"{row['Boundary_Separability_AUROC_Mean']:.3f} $\\pm$ "
                f"{row['Boundary_Separability_AUROC_SEM']:.3f} & "
                f"{row['Attention_Disagreement_Strength_Mean']:.3f} "
                f"$\\pm$ "
                f"{row['Attention_Disagreement_Strength_SEM']:.3f} \\\\\n"
            )

        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table*}\n")


def save_settings(
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Dataset={DATASET_NAME}\n")
        handle.write(f"L={L}\n")
        handle.write(f"Clean_CSV={CLEAN_CSV}\n")
        handle.write(f"Noisy_CSV={NOISY_CSV}\n")
        handle.write(f"Checkpoint={checkpoint_path}\n")
        handle.write(f"Seed={SEED}\n")
        handle.write(f"Tc={TC}\n")
        handle.write(f"Critical_width={CRITICAL_WIDTH}\n")
        handle.write(f"N_shuffles={N_SHUFFLES}\n")
        handle.write(
            "Boundary_definition=site differs from at least one of four "
            "nearest neighbours under periodic boundary conditions\n"
        )
        handle.write(
            "Local_disagreement=fraction of four nearest neighbours "
            "with a different spin\n"
        )
        handle.write(
            "Attention_map=raw sigmoid output of the trained spatial "
            "attention module\n"
        )
        handle.write(
            "Important_interpretation=the attention channel is concatenated "
            "with encoder features, so raw polarity is not assumed to mean "
            "importance. Both raw signed metrics and orientation-independent "
            "separability metrics are reported.\n"
        )
        handle.write(
            "Boundary_Contrast_Factor=max(R_A,1/R_A)\n"
        )
        handle.write(
            "Boundary_Separability_AUROC=max(raw_AUROC,1-raw_AUROC) "
            "for learned attention only\n"
        )
        handle.write(
            "Shuffled_control_AUROC=untransformed shuffled AUROC; expected "
            "approximately 0.5\n"
        )
        handle.write(
            "Attention_Disagreement_Strength=abs(raw_Pearson_correlation)\n"
        )


def main() -> None:
    set_seed(SEED)
    checkpoint_path = resolve_checkpoint()
    validate_inputs(checkpoint_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("REVIEWER 7: FINAL ORIENTATION-AWARE ATTENTION--PHYSICS ANALYSIS")
    print("=" * 78)
    print(f"Dataset       : {DATASET_NAME}")
    print(f"Clean CSV     : {CLEAN_CSV}")
    print(f"Noisy CSV     : {NOISY_CSV}")
    print(f"Checkpoint    : {checkpoint_path}")
    print(f"Output folder : {OUT_DIR}")
    print("Retraining    : No")

    device = resolve_device()

    (
        model_inputs,
        clean_spins,
        temperatures,
        phases,
        dataset_indices,
    ) = load_test_data(
        CLEAN_CSV,
        NOISY_CSV,
        L,
        SEED,
    )

    dataset = TestIsingDataset(
        model_inputs,
        clean_spins,
        temperatures,
        phases,
        dataset_indices,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(checkpoint_path, device)

    per_sample, examples = run_attention_analysis(
        model,
        loader,
        device,
    )
    regime_summary = build_regime_summary(per_sample)
    temperature_summary = build_temperature_summary(per_sample)

    per_sample_path = OUT_DIR / f"attention_per_sample_L{L}.csv"
    regime_path = OUT_DIR / f"attention_summary_regimes_L{L}.csv"
    temperature_path = OUT_DIR / f"attention_summary_temperature_L{L}.csv"
    curves_path = OUT_DIR / f"attention_physics_curves_L{L}.pdf"
    raw_curves_path = OUT_DIR / f"attention_raw_diagnostics_L{L}.pdf"
    examples_path = OUT_DIR / f"attention_examples_L{L}.pdf"
    latex_path = OUT_DIR / f"attention_table_L{L}.tex"
    settings_path = OUT_DIR / f"attention_settings_L{L}.txt"

    per_sample.to_csv(per_sample_path, index=False)
    regime_summary.to_csv(regime_path, index=False)
    temperature_summary.to_csv(temperature_path, index=False)

    save_attention_curves(temperature_summary, curves_path)
    save_raw_attention_diagnostics(temperature_summary, raw_curves_path)
    save_attention_examples(examples, examples_path)
    save_latex_table(regime_summary, latex_path)
    save_settings(checkpoint_path, settings_path)

    display_columns = [
        "Group",
        "N_samples",
        "Boundary_Response_Direction",
        "Boundary_Contrast_Factor_Mean",
        "Boundary_Separability_AUROC_Mean",
        "Shuffled_Separability_AUROC_Mean",
        "Attention_Disagreement_Strength_Mean",
        "Attention_Disagreement_Correlation_Mean",
    ]

    print("\nRegime-wise results")
    print("-" * 105)
    print(regime_summary[display_columns].to_string(index=False))
    print("-" * 105)

    print("\nOutputs")
    print(f"Per-sample CSV      : {per_sample_path}")
    print(f"Regime summary CSV  : {regime_path}")
    print(f"Temperature CSV     : {temperature_path}")
    print(f"Quantitative PDF    : {curves_path}")
    print(f"Raw diagnostics PDF : {raw_curves_path}")
    print(f"Example maps PDF    : {examples_path}")
    print(f"LaTeX table         : {latex_path}")
    print(f"Settings            : {settings_path}")

    print("\nReviewer 7 analysis completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Execution interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
