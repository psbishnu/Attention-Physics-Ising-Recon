#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file Restormer training, evaluation, and quantitative attention–physics
analysis for corrupted two-dimensional Ising spin configurations.

Edit only the USER SETTINGS section and run:

    python Restormer_Train_and_Attention_Analysis.py

Run one lattice size at a time by changing:

    L = 32
    L = 64
    L = 128

Workflow
--------
1. Load MCD{L}.csv and MCDN{L}.csv.
2. Use one deterministic 80/10/10 split.
3. If the Restormer checkpoint does not exist, train Restormer and save it.
4. If the checkpoint already exists, load it and skip training.
5. Report MAE_T, Acc_Phi, and ImpAcc.
6. Quantitatively analyze the final full-resolution Restormer attention response:
   - boundary versus interior attention;
   - boundary-enrichment ratio;
   - boundary AUROC;
   - spatially shuffled AUROC control;
   - correlation with local nearest-neighbour spin disagreement.
7. Save regime-wise and temperature-binned CSV files and PDFs.

Important interpretation
------------------------
Restormer MDTA is channel-wise transposed attention. Therefore, this script
analyzes the spatial magnitude of the output response of a full-resolution
attention block. In the manuscript, call it a "spatial attention-response map",
not a conventional spatial attention-probability matrix.
"""

from __future__ import annotations

import contextlib
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


# =============================================================================
# USER SETTINGS — EDIT ONLY THIS SECTION
# =============================================================================
L = 64

CLEAN_CSV = f"../JOB5_Noise/J5Data/MCD{L}.csv"
NOISY_CSV = f"../JOB5_Noise/J5Data/MCDN{L}.csv"

RESTORMER_OUT_DIR = f"./Restormer_L{L}"
CHECKPOINT = f"./Restormer_L{L}/best_restormer_L{L}.pth"
ATTENTION_OUT_DIR = f"./Attention_Physics_L{L}"

# The current PBS node shown by the user is CPU-only.
DEVICE = "cpu"                    # Change to "cuda" only on a GPU node.
USE_AMP = False                  # Keep False on CPU.
NUM_WORKERS = 0
CPU_THREADS = 16                 # Reduce if the scheduler allocates fewer cores.

# Conservative CPU batch sizes.
BATCH_SIZE = {32: 8, 64: 4, 128: 1}[L]

# Training.
EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 7
SEED = 123

# If checkpoint exists, it is loaded and training is skipped.
TRAIN_IF_CHECKPOINT_MISSING = True
FORCE_RETRAIN = False

# Loss weights.
LAMBDA_REC = 2.0
LAMBDA_MISSING = 2.0
LAMBDA_TEMPERATURE = 1.0
LAMBDA_PHASE = 1.0

# Compact Restormer architecture.
DIM = 32
NUM_BLOCKS = (2, 3, 3, 4)
NUM_REFINEMENT_BLOCKS = 2
HEADS = (1, 2, 4, 8)
FFN_EXPANSION_FACTOR = 2.66
LAYER_NORM_TYPE = "WithBias"

# Attention–physics analysis.
TC = 2.269
CRITICAL_WIDTH = 0.20
N_TEMPERATURE_BINS = 16
N_SHUFFLES = 10
ATTENTION_LAYER = "refinement_last"
# Alternative: ATTENTION_LAYER = "decoder_level1_last"
# =============================================================================


@dataclass(frozen=True)
class Config:
    lattice_size: int
    clean_csv: str
    noisy_csv: str
    restormer_out_dir: str
    checkpoint: str
    attention_out_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    seed: int
    num_workers: int
    device: str
    use_amp: bool
    lambda_rec: float
    lambda_missing: float
    lambda_temperature: float
    lambda_phase: float
    dim: int
    num_blocks: Tuple[int, int, int, int]
    num_refinement_blocks: int
    heads: Tuple[int, int, int, int]
    ffn_expansion_factor: float
    layer_norm_type: str


def build_config() -> Config:
    if L not in (32, 64, 128):
        raise ValueError("L must be 32, 64, or 128.")
    if DEVICE not in ("cpu", "cuda"):
        raise ValueError("DEVICE must be 'cpu' or 'cuda'.")
    if ATTENTION_LAYER not in ("refinement_last", "decoder_level1_last"):
        raise ValueError(
            "ATTENTION_LAYER must be 'refinement_last' or "
            "'decoder_level1_last'."
        )
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive.")

    return Config(
        lattice_size=L,
        clean_csv=CLEAN_CSV,
        noisy_csv=NOISY_CSV,
        restormer_out_dir=RESTORMER_OUT_DIR,
        checkpoint=CHECKPOINT,
        attention_out_dir=ATTENTION_OUT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        seed=SEED,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        use_amp=USE_AMP,
        lambda_rec=LAMBDA_REC,
        lambda_missing=LAMBDA_MISSING,
        lambda_temperature=LAMBDA_TEMPERATURE,
        lambda_phase=LAMBDA_PHASE,
        dim=DIM,
        num_blocks=tuple(NUM_BLOCKS),
        num_refinement_blocks=NUM_REFINEMENT_BLOCKS,
        heads=tuple(HEADS),
        ffn_expansion_factor=FFN_EXPANSION_FACTOR,
        layer_norm_type=LAYER_NORM_TYPE,
    )


# =============================================================================
# Reproducibility and data
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if CPU_THREADS > 0:
        torch.set_num_threads(CPU_THREADS)
        try:
            torch.set_num_interop_threads(max(1, min(4, CPU_THREADS)))
        except RuntimeError:
            pass


def extract_spin_columns(df: pd.DataFrame, lattice_size: int) -> List[str]:
    """Support spin_0... and Spin1... naming conventions."""
    candidates = [
        column for column in df.columns
        if column.lower().startswith("spin")
    ]
    if not candidates:
        raise ValueError(
            "No spin columns found. Expected names such as spin_0 or Spin1."
        )

    def suffix_index(name: str) -> int:
        digits = "".join(character for character in name if character.isdigit())
        return int(digits) if digits else 0

    candidates = sorted(candidates, key=suffix_index)
    expected = lattice_size * lattice_size
    if len(candidates) != expected:
        raise ValueError(
            f"Expected {expected} spin columns for L={lattice_size}, "
            f"but found {len(candidates)}."
        )
    return candidates


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
    mapped = normalized.map({"F": 0, "P": 1})
    if mapped.isna().any():
        unexpected = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"Unexpected phase labels: {unexpected}")
    return mapped.to_numpy(dtype=np.int64)


def load_data(config: Config) -> Tuple[np.ndarray, ...]:
    clean_path = Path(config.clean_csv)
    noisy_path = Path(config.noisy_csv)

    if not clean_path.exists():
        raise FileNotFoundError(f"Clean CSV not found: {clean_path.resolve()}")
    if not noisy_path.exists():
        raise FileNotFoundError(f"Noisy CSV not found: {noisy_path.resolve()}")

    clean_df = pd.read_csv(clean_path)
    noisy_df = pd.read_csv(noisy_path)

    if len(clean_df) != len(noisy_df):
        raise ValueError(
            f"Clean/noisy row counts differ: "
            f"{len(clean_df)} versus {len(noisy_df)}."
        )

    spin_columns = extract_spin_columns(clean_df, config.lattice_size)
    missing_in_noisy = [
        column for column in spin_columns
        if column not in noisy_df.columns
    ]
    if missing_in_noisy:
        raise ValueError(
            f"Noisy CSV is missing {len(missing_in_noisy)} spin columns."
        )

    lattice_size = config.lattice_size
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
    inputs = np.stack((noisy_spins, masks), axis=1).astype(np.float32)

    return inputs, clean_spins, temperatures, phases, noisy_spins, masks


class IsingDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        clean_spins: np.ndarray,
        temperatures: np.ndarray,
        phases: np.ndarray,
        noisy_spins: np.ndarray,
        masks: np.ndarray,
    ) -> None:
        self.inputs = torch.from_numpy(inputs)
        self.clean = torch.from_numpy(clean_spins)[:, None]
        self.temperature = torch.from_numpy(temperatures)[:, None]
        self.phase = torch.from_numpy(phases)
        self.noisy = torch.from_numpy(noisy_spins)[:, None]
        self.mask = torch.from_numpy(masks)[:, None]

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "dataset_index": torch.tensor(index, dtype=torch.long),
            "X": self.inputs[index],
            "S": self.clean[index],
            "T": self.temperature[index],
            "P": self.phase[index],
            "Y": self.noisy[index],
            "M": self.mask[index],
        }


def deterministic_split_indices(
    number_samples: int,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(
        number_samples,
        generator=generator,
    ).tolist()

    number_train = int(0.8 * number_samples)
    number_validation = int(0.1 * number_samples)

    train_indices = permutation[:number_train]
    validation_indices = permutation[
        number_train:number_train + number_validation
    ]
    test_indices = permutation[number_train + number_validation:]

    return train_indices, validation_indices, test_indices


def make_loaders(
    config: Config,
    dataset: Dataset,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_indices, validation_indices, test_indices = (
        deterministic_split_indices(len(dataset), config.seed)
    )

    pin_memory = (
        config.device == "cuda"
        and torch.cuda.is_available()
    )
    persistent_workers = config.num_workers > 0

    common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
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
# Restormer model
# =============================================================================
def to_3d(x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    return x.reshape(
        batch,
        channels,
        height * width,
    ).transpose(1, 2)


def to_4d(
    x: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    batch, sequence_length, channels = x.shape
    if sequence_length != height * width:
        raise ValueError(
            f"Cannot reshape sequence length {sequence_length} "
            f"to {height}x{width}."
        )
    return x.transpose(1, 2).reshape(
        batch,
        channels,
        height,
        width,
    )


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return (
            x / torch.sqrt(variance + 1e-5)
        ) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return (
            (x - mean) / torch.sqrt(variance + 1e-5)
        ) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    def __init__(
        self,
        dimension: int,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dimension)
        elif layer_norm_type == "WithBias":
            self.body = WithBiasLayerNorm(dimension)
        else:
            raise ValueError(
                "LAYER_NORM_TYPE must be BiasFree or WithBias."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        return to_4d(
            self.body(to_3d(x)),
            height,
            width,
        )


class FeedForward(nn.Module):
    """Gated-Dconv feed-forward network."""

    def __init__(
        self,
        dimension: int,
        expansion_factor: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(dimension * expansion_factor)

        self.project_in = nn.Conv2d(
            dimension,
            hidden * 2,
            kernel_size=1,
            bias=bias,
        )
        self.depthwise = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden * 2,
            bias=bias,
            padding_mode="circular",
        )
        self.project_out = nn.Conv2d(
            hidden,
            dimension,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.project_in(x)
        first, second = self.depthwise(projected).chunk(2, dim=1)
        return self.project_out(F.gelu(first) * second)


class Attention(nn.Module):
    """Multi-Dconv head transposed attention."""

    def __init__(
        self,
        dimension: int,
        number_heads: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dimension % number_heads != 0:
            raise ValueError(
                f"dimension={dimension} must be divisible by "
                f"number_heads={number_heads}."
            )

        self.number_heads = number_heads
        self.temperature = nn.Parameter(
            torch.ones(number_heads, 1, 1)
        )
        self.qkv = nn.Conv2d(
            dimension,
            dimension * 3,
            kernel_size=1,
            bias=bias,
        )
        self.qkv_depthwise = nn.Conv2d(
            dimension * 3,
            dimension * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dimension * 3,
            bias=bias,
            padding_mode="circular",
        )
        self.project_out = nn.Conv2d(
            dimension,
            dimension,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        query, key, value = self.qkv_depthwise(
            self.qkv(x)
        ).chunk(3, dim=1)

        channels_per_head = channels // self.number_heads

        query = query.reshape(
            batch,
            self.number_heads,
            channels_per_head,
            height * width,
        )
        key = key.reshape(
            batch,
            self.number_heads,
            channels_per_head,
            height * width,
        )
        value = value.reshape(
            batch,
            self.number_heads,
            channels_per_head,
            height * width,
        )

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        attention = (
            query @ key.transpose(-2, -1)
        ) * self.temperature
        attention = attention.softmax(dim=-1)

        output = attention @ value
        output = output.reshape(
            batch,
            channels,
            height,
            width,
        )
        return self.project_out(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        number_heads: int,
        expansion_factor: float,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dimension, layer_norm_type)
        self.attention = Attention(
            dimension,
            number_heads,
            bias=False,
        )
        self.norm2 = LayerNorm2d(dimension, layer_norm_type)
        self.feed_forward = FeedForward(
            dimension,
            expansion_factor,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dimension: int,
    ) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            input_channels,
            dimension,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            padding_mode="circular",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                padding_mode="circular",
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                padding_mode="circular",
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def make_blocks(
    dimension: int,
    count: int,
    heads: int,
    expansion_factor: float,
    layer_norm_type: str,
) -> nn.Sequential:
    return nn.Sequential(
        *[
            TransformerBlock(
                dimension=dimension,
                number_heads=heads,
                expansion_factor=expansion_factor,
                layer_norm_type=layer_norm_type,
            )
            for _ in range(count)
        ]
    )


class RestormerIsing(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        dimension = config.dim
        blocks = config.num_blocks
        heads = config.heads
        expansion = config.ffn_expansion_factor
        norm_type = config.layer_norm_type

        self.patch_embed = OverlapPatchEmbed(
            input_channels=2,
            dimension=dimension,
        )

        self.encoder_level1 = make_blocks(
            dimension,
            blocks[0],
            heads[0],
            expansion,
            norm_type,
        )
        self.down1_2 = Downsample(dimension)

        self.encoder_level2 = make_blocks(
            dimension * 2,
            blocks[1],
            heads[1],
            expansion,
            norm_type,
        )
        self.down2_3 = Downsample(dimension * 2)

        self.encoder_level3 = make_blocks(
            dimension * 4,
            blocks[2],
            heads[2],
            expansion,
            norm_type,
        )
        self.down3_4 = Downsample(dimension * 4)

        self.latent = make_blocks(
            dimension * 8,
            blocks[3],
            heads[3],
            expansion,
            norm_type,
        )

        self.up4_3 = Upsample(dimension * 8)
        self.reduce_channels3 = nn.Conv2d(
            dimension * 8,
            dimension * 4,
            kernel_size=1,
            bias=False,
        )
        self.decoder_level3 = make_blocks(
            dimension * 4,
            blocks[2],
            heads[2],
            expansion,
            norm_type,
        )

        self.up3_2 = Upsample(dimension * 4)
        self.reduce_channels2 = nn.Conv2d(
            dimension * 4,
            dimension * 2,
            kernel_size=1,
            bias=False,
        )
        self.decoder_level2 = make_blocks(
            dimension * 2,
            blocks[1],
            heads[1],
            expansion,
            norm_type,
        )

        self.up2_1 = Upsample(dimension * 2)
        self.decoder_level1 = make_blocks(
            dimension * 2,
            blocks[0],
            heads[0],
            expansion,
            norm_type,
        )
        self.refinement = make_blocks(
            dimension * 2,
            config.num_refinement_blocks,
            heads[0],
            expansion,
            norm_type,
        )

        self.spin_output = nn.Conv2d(
            dimension * 2,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            padding_mode="circular",
        )

        latent_dimension = dimension * 8
        self.temperature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(latent_dimension, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.phase_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(latent_dimension, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_level1 = self.patch_embed(x)
        output_level1 = self.encoder_level1(input_level1)

        input_level2 = self.down1_2(output_level1)
        output_level2 = self.encoder_level2(input_level2)

        input_level3 = self.down2_3(output_level2)
        output_level3 = self.encoder_level3(input_level3)

        input_level4 = self.down3_4(output_level3)
        latent = self.latent(input_level4)

        input_decoder3 = self.up4_3(latent)
        input_decoder3 = torch.cat(
            (input_decoder3, output_level3),
            dim=1,
        )
        input_decoder3 = self.reduce_channels3(input_decoder3)
        output_decoder3 = self.decoder_level3(input_decoder3)

        input_decoder2 = self.up3_2(output_decoder3)
        input_decoder2 = torch.cat(
            (input_decoder2, output_level2),
            dim=1,
        )
        input_decoder2 = self.reduce_channels2(input_decoder2)
        output_decoder2 = self.decoder_level2(input_decoder2)

        input_decoder1 = self.up2_1(output_decoder2)
        input_decoder1 = torch.cat(
            (input_decoder1, output_level1),
            dim=1,
        )
        output_decoder1 = self.decoder_level1(input_decoder1)
        output_decoder1 = self.refinement(output_decoder1)

        raw_spin = torch.tanh(
            self.spin_output(output_decoder1)
        )
        temperature = self.temperature_head(latent)
        phase_logits = self.phase_head(latent)

        return raw_spin, temperature, phase_logits


# =============================================================================
# Training utilities
# =============================================================================
def resolve_device(config: Config) -> torch.device:
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    if config.device == "cuda":
        print("CUDA requested but unavailable; falling back to CPU.")

    print("Using CPU")
    return torch.device("cpu")


def autocast_context(
    device: torch.device,
    enabled: bool,
):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return torch.cuda.amp.autocast(dtype=torch.float16)


def make_grad_scaler(
    device: torch.device,
    enabled: bool,
):
    use_scaler = enabled and device.type == "cuda"

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(
                "cuda",
                enabled=use_scaler,
            )
        except TypeError:
            return torch.amp.GradScaler(enabled=use_scaler)

    return torch.cuda.amp.GradScaler(enabled=use_scaler)


def move_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def compute_losses(
    config: Config,
    raw_spin: torch.Tensor,
    temperature_hat: torch.Tensor,
    phase_logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    phase_criterion: nn.Module,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    clean = batch["S"]
    temperature = batch["T"]
    phase = batch["P"]
    noisy = batch["Y"]
    mask = batch["M"]

    observed_loss = (
        mask * torch.abs(raw_spin - noisy)
    ).sum() / (mask.sum() + 1e-8)

    missing_mask = 1.0 - mask
    missing_loss = (
        missing_mask * torch.abs(raw_spin - clean)
    ).sum() / (missing_mask.sum() + 1e-8)

    reconstruction_loss = (
        observed_loss
        + config.lambda_missing * missing_loss
    )
    temperature_loss = torch.abs(
        temperature_hat - temperature
    ).mean()
    phase_loss = phase_criterion(
        phase_logits,
        phase,
    )

    total_loss = (
        config.lambda_rec * reconstruction_loss
        + config.lambda_temperature * temperature_loss
        + config.lambda_phase * phase_loss
    )

    components = {
        "total": total_loss,
        "reconstruction": reconstruction_loss,
        "observed": observed_loss,
        "missing": missing_loss,
        "temperature": temperature_loss,
        "phase": phase_loss,
    }
    return total_loss, components


def new_meter() -> Dict[str, float]:
    return {
        "total": 0.0,
        "reconstruction": 0.0,
        "observed": 0.0,
        "missing": 0.0,
        "temperature": 0.0,
        "phase": 0.0,
    }


def update_meter(
    meter: Dict[str, float],
    components: Dict[str, torch.Tensor],
) -> None:
    for name in meter:
        meter[name] += float(
            components[name].detach().item()
        )


def average_meter(
    meter: Dict[str, float],
    count: int,
) -> Dict[str, float]:
    divisor = max(1, count)
    return {
        name: value / divisor
        for name, value in meter.items()
    }


def run_training_epoch(
    config: Config,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    phase_criterion: nn.Module,
    scaler,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    meter = new_meter()
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:03d} [train]",
        leave=False,
        ncols=110,
    )

    for batch in progress:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, config.use_amp):
            raw_spin, temperature_hat, phase_logits = model(
                batch["X"]
            )
            total_loss, components = compute_losses(
                config,
                raw_spin,
                temperature_hat,
                phase_logits,
                batch,
                phase_criterion,
            )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        update_meter(meter, components)

        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                loss=f"{components['total'].item():.4f}",
                rec=f"{components['reconstruction'].item():.4f}",
            )

    return average_meter(meter, len(loader))


@torch.no_grad()
def run_validation_epoch(
    config: Config,
    model: nn.Module,
    loader: DataLoader,
    phase_criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    meter = new_meter()

    for batch in loader:
        batch = move_batch(batch, device)

        with autocast_context(device, config.use_amp):
            raw_spin, temperature_hat, phase_logits = model(
                batch["X"]
            )
            _, components = compute_losses(
                config,
                raw_spin,
                temperature_hat,
                phase_logits,
                batch,
                phase_criterion,
            )

        update_meter(meter, components)

    return average_meter(meter, len(loader))


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    config: Config,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": validation_loss,
            "config": asdict(config),
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Dict:
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=device,
        )


def train_model(
    config: Config,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[Dict[str, float]], int, float]:
    output_dir = Path(config.restormer_out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config.checkpoint)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6,
    )
    phase_criterion = nn.CrossEntropyLoss()
    scaler = make_grad_scaler(
        device,
        config.use_amp,
    )

    best_validation_loss = math.inf
    best_epoch = 0
    patience_counter = 0
    history: List[Dict[str, float]] = []
    training_start = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_training_epoch(
            config,
            model,
            train_loader,
            optimizer,
            phase_criterion,
            scaler,
            device,
            epoch,
        )
        validation_metrics = run_validation_epoch(
            config,
            model,
            validation_loader,
            phase_criterion,
            device,
        )
        scheduler.step()

        epoch_seconds = time.time() - epoch_start
        row: Dict[str, float] = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": epoch_seconds,
        }
        row.update({
            f"train_{key}": value
            for key, value in train_metrics.items()
        })
        row.update({
            f"val_{key}": value
            for key, value in validation_metrics.items()
        })
        history.append(row)

        improved = (
            validation_metrics["total"]
            < best_validation_loss
        )
        if improved:
            best_validation_loss = validation_metrics["total"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                best_validation_loss,
                config,
            )
        else:
            patience_counter += 1

        marker = "*" if improved else " "
        print(
            f"{marker} Epoch {epoch:03d}/{config.epochs} | "
            f"train={train_metrics['total']:.5f} | "
            f"val={validation_metrics['total']:.5f} | "
            f"val_rec={validation_metrics['reconstruction']:.5f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"{epoch_seconds:.1f}s | "
            f"patience={patience_counter}/{config.patience}"
        )

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    training_seconds = time.time() - training_start
    print(
        f"Training completed in "
        f"{training_seconds / 60.0:.2f} minutes."
    )

    history_path = output_dir / (
        f"restormer_training_history_L"
        f"{config.lattice_size}.csv"
    )
    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    if not checkpoint_path.exists():
        raise RuntimeError(
            "Training completed without creating a checkpoint."
        )

    checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return (
        history,
        int(checkpoint.get("epoch", best_epoch)),
        float(checkpoint.get(
            "val_loss",
            best_validation_loss,
        )),
    )


# =============================================================================
# Restormer evaluation
# =============================================================================
@torch.no_grad()
def evaluate_restormer(
    config: Config,
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()

    absolute_temperature_error = 0.0
    correct_phase = 0
    number_samples = 0
    correct_missing = 0
    number_missing = 0
    prediction_rows: List[Dict[str, float]] = []

    for batch in tqdm(
        test_loader,
        desc="Restormer test evaluation",
        ncols=110,
    ):
        dataset_indices = batch["dataset_index"].numpy()
        batch = move_batch(batch, device)

        with autocast_context(device, config.use_amp):
            raw_spin, temperature_hat, phase_logits = model(
                batch["X"]
            )

        final_spin = (
            batch["M"] * batch["Y"]
            + (1.0 - batch["M"]) * raw_spin
        )
        binary_spin = torch.where(
            final_spin >= 0,
            1.0,
            -1.0,
        )
        phase_prediction = phase_logits.argmax(dim=1)

        batch_size = batch["X"].shape[0]
        absolute_temperature_error += float(
            torch.abs(
                temperature_hat - batch["T"]
            ).sum().item()
        )
        correct_phase += int(
            (phase_prediction == batch["P"]).sum().item()
        )
        number_samples += batch_size

        missing = batch["M"] == 0
        correct_missing += int(
            (
                (binary_spin == batch["S"])
                & missing
            ).sum().item()
        )
        number_missing += int(missing.sum().item())

        true_temperature = (
            batch["T"].squeeze(1).cpu().numpy()
        )
        predicted_temperature = (
            temperature_hat.squeeze(1)
            .float()
            .cpu()
            .numpy()
        )
        true_phase = batch["P"].cpu().numpy()
        predicted_phase = phase_prediction.cpu().numpy()

        for index in range(batch_size):
            prediction_rows.append(
                {
                    "DatasetIndex": int(
                        dataset_indices[index]
                    ),
                    "T_true": float(
                        true_temperature[index]
                    ),
                    "T_pred": float(
                        predicted_temperature[index]
                    ),
                    "Phase_true": int(
                        true_phase[index]
                    ),
                    "Phase_pred": int(
                        predicted_phase[index]
                    ),
                }
            )

    if number_samples == 0:
        raise RuntimeError("The test loader is empty.")
    if number_missing == 0:
        raise RuntimeError(
            "The test set contains no missing sites."
        )

    metrics = {
        "Method": "Restormer",
        "L": config.lattice_size,
        "MAE_T": (
            absolute_temperature_error
            / number_samples
        ),
        "Acc_Phi": correct_phase / number_samples,
        "ImpAcc": correct_missing / number_missing,
        "N_test": number_samples,
        "N_missing_test": number_missing,
    }
    return metrics, pd.DataFrame(prediction_rows)


def save_restormer_outputs(
    config: Config,
    metrics: Dict[str, float],
    predictions: pd.DataFrame,
    best_epoch: int,
    best_validation_loss: float,
    parameter_count: int,
) -> None:
    output_dir = Path(config.restormer_out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    complete_metrics = dict(metrics)
    complete_metrics.update(
        {
            "Best_Epoch": best_epoch,
            "Best_Val_Loss": best_validation_loss,
            "Trainable_Parameters": parameter_count,
            "Seed": config.seed,
        }
    )

    metrics_path = output_dir / (
        f"restormer_metrics_L"
        f"{config.lattice_size}.csv"
    )
    predictions_path = output_dir / (
        f"restormer_predictions_L"
        f"{config.lattice_size}.csv"
    )
    config_path = output_dir / (
        f"restormer_config_L"
        f"{config.lattice_size}.txt"
    )

    pd.DataFrame([complete_metrics]).to_csv(
        metrics_path,
        index=False,
    )
    predictions.to_csv(
        predictions_path,
        index=False,
    )

    with config_path.open("w", encoding="utf-8") as handle:
        for key, value in asdict(config).items():
            handle.write(f"{key}={value}\n")

    print("\nRestormer test results")
    print("-" * 60)
    print(f"MAE_T   : {complete_metrics['MAE_T']:.6f}")
    print(f"Acc_Phi : {complete_metrics['Acc_Phi']:.6f}")
    print(f"ImpAcc  : {complete_metrics['ImpAcc']:.6f}")
    print("-" * 60)
    print(f"Metrics     : {metrics_path}")
    print(f"Predictions : {predictions_path}")


# =============================================================================
# Physical attention maps and metrics
# =============================================================================
def local_disagreement_map(
    spins: np.ndarray,
) -> np.ndarray:
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


def domain_boundary_map(
    spins: np.ndarray,
) -> np.ndarray:
    return local_disagreement_map(spins) > 0.0


def normalize_attention_response(
    response: torch.Tensor,
) -> torch.Tensor:
    spatial = response.float().abs().mean(dim=1)
    flattened = spatial.flatten(start_dim=1)
    minimum = flattened.min(dim=1).values[:, None, None]
    maximum = flattened.max(dim=1).values[:, None, None]

    return (
        spatial - minimum
    ) / (maximum - minimum + 1e-8)


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        average_rank = 0.5 * (
            (start + 1) + end
        )
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def binary_roc_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    labels = np.asarray(
        labels,
        dtype=np.int64,
    ).reshape(-1)
    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

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
    first = np.asarray(
        first,
        dtype=np.float64,
    ).reshape(-1)
    second = np.asarray(
        second,
        dtype=np.float64,
    ).reshape(-1)

    if (
        first.size < 2
        or np.std(first) < 1e-12
        or np.std(second) < 1e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(first, second)[0, 1]
    )


def temperature_regime(
    temperature: float,
) -> str:
    if temperature < TC - CRITICAL_WIDTH:
        return "Ordered"
    if temperature > TC + CRITICAL_WIDTH:
        return "Disordered"
    return "Near-critical"


def analyze_attention_sample(
    attention: np.ndarray,
    clean_spin: np.ndarray,
    random_generator: np.random.Generator,
) -> Dict[str, float]:
    disagreement = local_disagreement_map(clean_spin)
    boundary = disagreement > 0.0
    interior = ~boundary

    mean_boundary = (
        float(attention[boundary].mean())
        if boundary.any()
        else float("nan")
    )
    mean_interior = (
        float(attention[interior].mean())
        if interior.any()
        else float("nan")
    )

    enrichment_ratio = (
        mean_boundary / (mean_interior + 1e-12)
        if (
            np.isfinite(mean_boundary)
            and np.isfinite(mean_interior)
        )
        else float("nan")
    )

    labels = boundary.astype(np.int64).reshape(-1)
    flattened_attention = attention.reshape(-1)

    boundary_auc = binary_roc_auc(
        labels,
        flattened_attention,
    )
    correlation = safe_pearson(
        flattened_attention,
        disagreement.reshape(-1),
    )

    shuffled_aucs = []
    for _ in range(max(1, N_SHUFFLES)):
        shuffled = random_generator.permutation(
            flattened_attention
        )
        shuffled_aucs.append(
            binary_roc_auc(labels, shuffled)
        )

    return {
        "MeanAttention_Boundary": mean_boundary,
        "MeanAttention_Interior": mean_interior,
        "BoundaryEnrichment_Ratio": enrichment_ratio,
        "Boundary_AUROC": boundary_auc,
        "Shuffled_AUROC": float(
            np.nanmean(shuffled_aucs)
        ),
        "Attention_Disagreement_Correlation": correlation,
        "BoundaryFraction": float(boundary.mean()),
        "MeanLocalDisagreement": float(
            disagreement.mean()
        ),
    }


def select_attention_module(
    model: RestormerIsing,
) -> nn.Module:
    if ATTENTION_LAYER == "refinement_last":
        if len(model.refinement) == 0:
            raise ValueError(
                "The model has no refinement blocks."
            )
        return model.refinement[-1].attention

    if ATTENTION_LAYER == "decoder_level1_last":
        if len(model.decoder_level1) == 0:
            raise ValueError(
                "The model has no decoder-level-1 blocks."
            )
        return model.decoder_level1[-1].attention

    raise ValueError(
        f"Unsupported ATTENTION_LAYER: {ATTENTION_LAYER}"
    )


class AttentionResponseCapture:
    def __init__(self, module: nn.Module) -> None:
        self.output = None
        self.handle = module.register_forward_hook(
            self._hook
        )

    def _hook(
        self,
        _module,
        _inputs,
        output,
    ) -> None:
        self.output = output.detach()

    def close(self) -> None:
        self.handle.remove()


ATTENTION_METRIC_COLUMNS = [
    "MeanAttention_Boundary",
    "MeanAttention_Interior",
    "BoundaryEnrichment_Ratio",
    "Boundary_AUROC",
    "Shuffled_AUROC",
    "Attention_Disagreement_Correlation",
    "BoundaryFraction",
    "MeanLocalDisagreement",
]


def summarize_attention_group(
    data: pd.DataFrame,
    label: str,
) -> Dict[str, float]:
    row: Dict[str, float] = {
        "Group": label,
        "N_samples": int(len(data)),
        "Mean_Temperature": float(
            data["Temperature"].mean()
        ),
    }

    for column in ATTENTION_METRIC_COLUMNS:
        values = data[column].to_numpy(
            dtype=np.float64
        )
        finite_count = int(
            np.isfinite(values).sum()
        )

        row[f"{column}_Mean"] = float(
            np.nanmean(values)
        )
        row[f"{column}_Std"] = (
            float(np.nanstd(values, ddof=1))
            if finite_count > 1
            else 0.0
        )
        row[f"{column}_SEM"] = (
            float(
                np.nanstd(values, ddof=1)
                / math.sqrt(finite_count)
            )
            if finite_count > 1
            else 0.0
        )

    return row


def build_regime_summary(
    per_sample: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        summarize_attention_group(
            per_sample,
            "Overall",
        )
    ]

    for regime in (
        "Ordered",
        "Near-critical",
        "Disordered",
    ):
        subset = per_sample[
            per_sample["Regime"] == regime
        ]
        if len(subset) > 0:
            rows.append(
                summarize_attention_group(
                    subset,
                    regime,
                )
            )

    return pd.DataFrame(rows)


def build_temperature_bin_summary(
    per_sample: pd.DataFrame,
) -> pd.DataFrame:
    minimum = float(
        per_sample["Temperature"].min()
    )
    maximum = float(
        per_sample["Temperature"].max()
    )
    edges = np.linspace(
        minimum,
        maximum,
        N_TEMPERATURE_BINS + 1,
    )
    centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    bin_ids = np.digitize(
        per_sample["Temperature"].to_numpy(),
        edges,
    ) - 1
    bin_ids = np.clip(
        bin_ids,
        0,
        N_TEMPERATURE_BINS - 1,
    )

    rows = []
    for bin_id in range(N_TEMPERATURE_BINS):
        subset = per_sample.iloc[
            np.where(bin_ids == bin_id)[0]
        ]
        if len(subset) == 0:
            continue

        row = summarize_attention_group(
            subset,
            f"Bin_{bin_id + 1}",
        )
        row["T_center"] = float(centers[bin_id])
        row["T_min"] = float(edges[bin_id])
        row["T_max"] = float(edges[bin_id + 1])
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("T_center")
        .reset_index(drop=True)
    )


def save_attention_curves(
    binned: pd.DataFrame,
    output_path: Path,
) -> None:
    with PdfPages(output_path) as pdf:
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(7.5, 11.0),
            sharex=True,
        )

        axes[0].errorbar(
            binned["T_center"],
            binned["BoundaryEnrichment_Ratio_Mean"],
            yerr=binned[
                "BoundaryEnrichment_Ratio_SEM"
            ],
            marker="o",
            capsize=3,
            label="Learned attention response",
        )
        axes[0].axhline(
            1.0,
            linestyle="--",
            label="No enrichment",
        )
        axes[0].axvline(TC, linestyle=":")
        axes[0].set_ylabel(r"$R_A$")
        axes[0].set_title(
            "Boundary-attention enrichment"
        )
        axes[0].legend()

        axes[1].errorbar(
            binned["T_center"],
            binned["Boundary_AUROC_Mean"],
            yerr=binned["Boundary_AUROC_SEM"],
            marker="o",
            capsize=3,
            label="Learned response",
        )
        axes[1].plot(
            binned["T_center"],
            binned["Shuffled_AUROC_Mean"],
            marker="s",
            label="Shuffled control",
        )
        axes[1].axhline(
            0.5,
            linestyle="--",
            label="Random AUROC",
        )
        axes[1].axvline(TC, linestyle=":")
        axes[1].set_ylabel("Boundary AUROC")
        axes[1].set_title("Boundary localization")
        axes[1].legend()

        axes[2].errorbar(
            binned["T_center"],
            binned[
                "Attention_Disagreement_Correlation_Mean"
            ],
            yerr=binned[
                "Attention_Disagreement_Correlation_SEM"
            ],
            marker="o",
            capsize=3,
        )
        axes[2].axhline(0.0, linestyle="--")
        axes[2].axvline(TC, linestyle=":")
        axes[2].set_xlabel("Temperature")
        axes[2].set_ylabel(r"$\rho_{A,D}$")
        axes[2].set_title(
            "Attention–local-disagreement correlation"
        )

        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)


def save_attention_examples(
    examples: List[Dict],
    output_path: Path,
) -> None:
    if not examples:
        return

    with PdfPages(output_path) as pdf:
        for example in examples:
            figure, axes = plt.subplots(
                1,
                4,
                figsize=(13.0, 3.4),
            )

            axes[0].imshow(
                example["clean_spin"],
                vmin=-1,
                vmax=1,
                cmap="gray",
            )
            axes[0].set_title("Clean lattice")
            axes[0].axis("off")

            axes[1].imshow(
                example["boundary"],
                vmin=0,
                vmax=1,
                cmap="gray",
            )
            axes[1].set_title("Domain boundary")
            axes[1].axis("off")

            attention_image = axes[2].imshow(
                example["attention"],
                vmin=0,
                vmax=1,
            )
            axes[2].set_title("Attention response")
            axes[2].axis("off")
            figure.colorbar(
                attention_image,
                ax=axes[2],
                fraction=0.046,
                pad=0.04,
            )

            disagreement_image = axes[3].imshow(
                example["disagreement"],
                vmin=0,
                vmax=1,
            )
            axes[3].set_title("Local disagreement")
            axes[3].axis("off")
            figure.colorbar(
                disagreement_image,
                ax=axes[3],
                fraction=0.046,
                pad=0.04,
            )

            figure.suptitle(
                f"L={example['L']}, "
                f"T={example['temperature']:.3f}, "
                f"{example['regime']}, "
                f"R_A={example['enrichment']:.3f}, "
                f"AUROC={example['auc']:.3f}"
            )
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)


@torch.no_grad()
def run_attention_analysis(
    config: Config,
    model: RestormerIsing,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    output_dir = Path(config.attention_out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_attention = select_attention_module(model)
    capture = AttentionResponseCapture(target_attention)
    random_generator = np.random.default_rng(
        config.seed
    )

    per_sample_rows: List[Dict[str, float]] = []
    example_candidates: List[Dict] = []

    model.eval()

    try:
        for batch in tqdm(
            test_loader,
            desc="Attention–physics analysis",
            ncols=110,
        ):
            dataset_indices = (
                batch["dataset_index"].numpy()
            )
            clean_batch = batch["S"][:, 0].numpy()
            temperature_batch = (
                batch["T"][:, 0].numpy()
            )
            phase_batch = batch["P"].numpy()

            inputs = batch["X"].to(
                device,
                non_blocking=True,
            )

            with autocast_context(
                device,
                config.use_amp,
            ):
                model(inputs)

            if capture.output is None:
                raise RuntimeError(
                    "Attention hook did not capture output."
                )

            attention_batch = (
                normalize_attention_response(
                    capture.output
                )
                .cpu()
                .numpy()
            )

            for index in range(len(clean_batch)):
                attention = attention_batch[index]
                clean_spin = clean_batch[index]
                temperature = float(
                    temperature_batch[index]
                )
                regime = temperature_regime(
                    temperature
                )

                sample_metrics = analyze_attention_sample(
                    attention,
                    clean_spin,
                    random_generator,
                )

                row = {
                    "L": config.lattice_size,
                    "DatasetIndex": int(
                        dataset_indices[index]
                    ),
                    "Temperature": temperature,
                    "Phase": int(phase_batch[index]),
                    "Regime": regime,
                }
                row.update(sample_metrics)
                per_sample_rows.append(row)

                example_candidates.append(
                    {
                        "L": config.lattice_size,
                        "temperature": temperature,
                        "regime": regime,
                        "clean_spin": clean_spin.copy(),
                        "boundary": (
                            domain_boundary_map(
                                clean_spin
                            ).astype(float)
                        ),
                        "disagreement": (
                            local_disagreement_map(
                                clean_spin
                            )
                        ),
                        "attention": attention.copy(),
                        "enrichment": sample_metrics[
                            "BoundaryEnrichment_Ratio"
                        ],
                        "auc": sample_metrics[
                            "Boundary_AUROC"
                        ],
                    }
                )
    finally:
        capture.close()

    if not per_sample_rows:
        raise RuntimeError(
            "No test samples were analyzed."
        )

    per_sample = (
        pd.DataFrame(per_sample_rows)
        .sort_values(
            ["Temperature", "DatasetIndex"]
        )
        .reset_index(drop=True)
    )

    regime_summary = build_regime_summary(
        per_sample
    )
    temperature_summary = (
        build_temperature_bin_summary(
            per_sample
        )
    )

    lattice_size = config.lattice_size
    per_sample_path = output_dir / (
        f"attention_per_sample_L"
        f"{lattice_size}.csv"
    )
    regime_path = output_dir / (
        f"attention_summary_regimes_L"
        f"{lattice_size}.csv"
    )
    temperature_path = output_dir / (
        f"attention_summary_temperature_bins_L"
        f"{lattice_size}.csv"
    )
    curve_path = output_dir / (
        f"attention_physics_curves_L"
        f"{lattice_size}.pdf"
    )
    example_path = output_dir / (
        f"attention_examples_L"
        f"{lattice_size}.pdf"
    )
    settings_path = output_dir / (
        f"analysis_settings_L"
        f"{lattice_size}.txt"
    )

    per_sample.to_csv(
        per_sample_path,
        index=False,
    )
    regime_summary.to_csv(
        regime_path,
        index=False,
    )
    temperature_summary.to_csv(
        temperature_path,
        index=False,
    )

    save_attention_curves(
        temperature_summary,
        curve_path,
    )

    targets = [1.0, 2.0, TC, 4.0]
    selected_examples = []
    used_indices = set()

    for target in targets:
        available = [
            (index, item)
            for index, item in enumerate(
                example_candidates
            )
            if index not in used_indices
        ]
        if not available:
            break

        selected_index, selected_item = min(
            available,
            key=lambda pair: abs(
                pair[1]["temperature"] - target
            ),
        )
        used_indices.add(selected_index)
        selected_examples.append(selected_item)

    save_attention_examples(
        selected_examples,
        example_path,
    )

    with settings_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(f"L={lattice_size}\n")
        handle.write(
            f"checkpoint={config.checkpoint}\n"
        )
        handle.write(
            f"clean_csv={config.clean_csv}\n"
        )
        handle.write(
            f"noisy_csv={config.noisy_csv}\n"
        )
        handle.write(f"seed={config.seed}\n")
        handle.write(f"Tc={TC}\n")
        handle.write(
            f"critical_width={CRITICAL_WIDTH}\n"
        )
        handle.write(
            f"n_temperature_bins="
            f"{N_TEMPERATURE_BINS}\n"
        )
        handle.write(
            f"n_shuffles={N_SHUFFLES}\n"
        )
        handle.write(
            f"attention_layer={ATTENTION_LAYER}\n"
        )
        handle.write(
            "attention_definition="
            "mean absolute full-resolution output response "
            "of the selected Restormer attention block, "
            "normalized separately for each sample\n"
        )

    display_columns = [
        "Group",
        "N_samples",
        "BoundaryEnrichment_Ratio_Mean",
        "Boundary_AUROC_Mean",
        "Shuffled_AUROC_Mean",
        "Attention_Disagreement_Correlation_Mean",
    ]

    print("\nRegime-wise attention results")
    print("-" * 100)
    print(
        regime_summary[
            display_columns
        ].to_string(index=False)
    )
    print("-" * 100)

    print("\nAttention outputs")
    print(f"Per-sample CSV : {per_sample_path}")
    print(f"Regime CSV     : {regime_path}")
    print(f"Temperature CSV: {temperature_path}")
    print(f"Curves PDF     : {curve_path}")
    print(f"Examples PDF   : {example_path}")


# =============================================================================
# Main combined workflow
# =============================================================================
def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    config = build_config()
    set_seed(config.seed)

    restormer_output = Path(
        config.restormer_out_dir
    )
    attention_output = Path(
        config.attention_out_dir
    )
    checkpoint_path = Path(config.checkpoint)

    restormer_output.mkdir(
        parents=True,
        exist_ok=True,
    )
    attention_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("RESTORMER TRAINING + ATTENTION–PHYSICS ANALYSIS")
    print("=" * 78)
    print(f"Lattice size       : {config.lattice_size}")
    print(f"Clean CSV          : {config.clean_csv}")
    print(f"Noisy CSV          : {config.noisy_csv}")
    print(f"Checkpoint         : {config.checkpoint}")
    print(f"Restormer output   : {config.restormer_out_dir}")
    print(f"Attention output   : {config.attention_out_dir}")
    print(f"Requested device   : {config.device}")
    print(f"Batch size         : {config.batch_size}")
    print(f"CPU threads        : {CPU_THREADS}")

    device = resolve_device(config)

    (
        inputs,
        clean_spins,
        temperatures,
        phases,
        noisy_spins,
        masks,
    ) = load_data(config)

    dataset = IsingDataset(
        inputs,
        clean_spins,
        temperatures,
        phases,
        noisy_spins,
        masks,
    )
    (
        train_loader,
        validation_loader,
        test_loader,
    ) = make_loaders(config, dataset)

    print(
        f"Samples: total={len(dataset)}, "
        f"train={len(train_loader.dataset)}, "
        f"validation={len(validation_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )

    model = RestormerIsing(config).to(device)
    parameter_count = count_parameters(model)
    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    should_train = (
        FORCE_RETRAIN
        or not checkpoint_path.exists()
    )

    if should_train:
        if (
            not checkpoint_path.exists()
            and not TRAIN_IF_CHECKPOINT_MISSING
        ):
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path.resolve()}"
            )

        if FORCE_RETRAIN:
            print(
                "\nFORCE_RETRAIN=True: training a new model."
            )
        else:
            print(
                "\nCheckpoint not found. "
                "Training Restormer first."
            )

        (
            _history,
            best_epoch,
            best_validation_loss,
        ) = train_model(
            config,
            model,
            train_loader,
            validation_loader,
            device,
        )
    else:
        print(
            "\nCheckpoint found. "
            "Skipping training and loading it."
        )
        checkpoint = load_checkpoint(
            checkpoint_path,
            device,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        best_epoch = int(
            checkpoint.get("epoch", -1)
        )
        best_validation_loss = float(
            checkpoint.get("val_loss", float("nan"))
        )

    model.eval()

    metrics, predictions = evaluate_restormer(
        config,
        model,
        test_loader,
        device,
    )
    save_restormer_outputs(
        config,
        metrics,
        predictions,
        best_epoch,
        best_validation_loss,
        parameter_count,
    )

    run_attention_analysis(
        config,
        model,
        test_loader,
        device,
    )

    print("\n" + "=" * 78)
    print("COMBINED WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "Execution interrupted by user.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        raise
