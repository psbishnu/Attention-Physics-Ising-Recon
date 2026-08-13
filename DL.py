# -*- coding: utf-8 -*-

# Program 3 (ENHANCED): Physics-informed deep model for inverse inference
# on MCD{L}.csv (clean) + MCDN{L}.csv (noisy).
# - ENHANCED with advanced noise removal and improved reconstruction
# - Loads CSVs (Temperature, Phase, spin_* columns)
# - Builds tensors: X=[Y,Mask], targets=(Sigma, T, Phase)
# - Trains CNN with enhanced architecture and loss functions
# - Saves CSVs + FOUR PDFs in black & white with bold, larger fonts:
#
#  1) physics_plots.pdf
#     - Page 1: |M|(T), E(T), C1(T), C_v(T) in one 2×2 frame
#     - Page 2: Energy comparison scatter (Clean vs Recon, Clean vs Noisy)
#
#  2) training_plots.pdf
#     - Training & validation loss
#
#  3) configs_bw.pdf
#     - Single page, 4 rows (T˜1,2,Tc,4) × 4 columns
#       (Clean | Noisy | Attention | Reconstructed)
#
#  4) accuracy_vs_temperature.pdf
#     - Reconstruction Accuracy vs Temperature
#
# Also writes:
#    metrics_global.csv
#    metrics_per_temp_bin.csv
#    preds_test.csv
#    training_times.csv

# ==========================
# ENHANCED USER PARAMETERS 19_12_2025
# ==========================

from pathlib import Path

L = 128
Tc = 2.269


# NOTE: match your Wolff generator file names
CLEAN_CSV = Path(f"../JOB5_Noise/J5Data/MCD128.csv")
NOISY_CSV = Path(f"../JOB5_Noise/J5Data/MCDN128.csv")
OUT_DIR   = "./Generated_L128"  # where to save pdf/csvs
EPOCHS    = 40
BATCH_SIZE = 64
LR         = 3e-4
DEVICE     = "cuda"   # "cuda" or "cpu"

# ENHANCED Loss weights with focus on reconstruction
# C1 is retained for evaluation; C2 is used as the independent correlation loss.
LAM_REC   = 2.0
LAM_E     = 0.1
LAM_M     = 0.1
LAM_C2    = 0.1  # independent longer-range correlation regularizer
LAM_T     = 1.0
LAM_PHASE = 1.0
LAM_BIN   = 0.5

# Temperature binning param (no longer used for Cv, but kept for compatibility)
N_BINS    = 16  # bins across [1,4]

# Early stopping
PATIENCE = 7

# Random seed
SEED = 123

# ==========================
# Imports
# ==========================
import os
import time
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from tqdm.auto import tqdm

# Start timing
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)

print(f"=== ENHANCED Physics-Informed Ising Model Reconstruction ===")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {OUT_DIR}")

# ==========================
# Helper: robust phase + spin-column handling
# ==========================
def _extract_spin_columns(df, L):
    """
    Support both legacy 'Spin1..SpinNN' and new 'spin_0..spin_NN-1' patterns.
    Returns a sorted list of spin-column names.
    """
    # any column except Temperature, Phase is candidate spin
    spin_cands = [c for c in df.columns if c.lower().startswith("spin")]
    if not spin_cands:
        raise ValueError("No spin columns found (expected columns like 'Spin1' or 'spin_0').")
    # sort by numeric suffix
    def _suffix_idx(name):
        # handle 'Spin1', 'Spin10', 'spin_0', 'spin_15', etc.
        s = "".join(ch for ch in name if ch.isdigit())
        return int(s) if s != "" else 0
    spin_cands_sorted = sorted(spin_cands, key=_suffix_idx)

    n_spins = len(spin_cands_sorted)
    if n_spins != L * L:
        raise ValueError(f"Expected {L*L} spin columns, got {n_spins}. Check L or CSV format.")
    return spin_cands_sorted

def _phase_to_int(series):
    """
    Map Phase to {0,1} as int64.
    - If already numeric, just astype(int64).
    - If 'F'/'P' strings, map F->1, P->0 (ferro vs para).
    """
    if series.dtype == np.int64 or series.dtype == np.int32 or np.issubdtype(series.dtype, np.number):
        return series.astype(np.int64)
    # assume strings like 'F'/'P'
    mapping = {'F': 1, 'P': 0, 'f': 1, 'p': 0}
    mapped = series.map(mapping)
    if mapped.isna().any():
        raise ValueError(f"Phase column contains unexpected values: {series.unique()}")
    return mapped.astype(np.int64)

# ==========================
# Enhanced Data loading with progress
# ==========================
def load_csvs_with_progress(clean_path, noisy_path, L):
    """Load datasets with progress tracking, robust to Phase format and spin_* names."""
    print("?? Loading datasets...")
    
    with tqdm(total=4, desc="Loading data") as pbar:
        df_clean = pd.read_csv(clean_path)
        pbar.update(1); pbar.set_description("Loaded clean data")
        
        df_noisy = pd.read_csv(noisy_path)
        pbar.update(1); pbar.set_description("Loaded noisy data")
        
        # robust spin-column detection
        spin_cols = _extract_spin_columns(df_clean, L)
        
        T  = df_clean["Temperature"].to_numpy(dtype=np.float32)
        Ph = _phase_to_int(df_clean["Phase"]).to_numpy(dtype=np.int64)
        pbar.update(1); pbar.set_description("Processed metadata")
        
        Sigma = df_clean[spin_cols].to_numpy(dtype=np.int8).reshape(-1, L, L)
        Y     = df_noisy[spin_cols].to_numpy(dtype=np.int8).reshape(-1, L, L)
        Mask  = (Y != 0).astype(np.float32)
        pbar.update(1); pbar.set_description("Processed spin data")
    
    # Model inputs
    X = np.stack([Y.astype(np.float32), Mask], axis=1)
    Sigma = Sigma.astype(np.float32)
    
    return X, Sigma, T, Ph, Y.astype(np.float32), Mask

data_load_start = time.time()
X, Sigma, T, Phase, Y, Mask = load_csvs_with_progress(CLEAN_CSV, NOISY_CSV, L)
N = X.shape[0]
data_load_time = time.time() - data_load_start
print(f"? Loaded {N:,} samples. Shapes: X={X.shape}, Sigma={Sigma.shape}")
print(f"??  Data loading time: {data_load_time:.2f}s")

class IsingCSVDataset(Dataset):
    def __init__(self, X, Sigma, T, Phase, Y, Mask):
        self.X = torch.from_numpy(X)                    # (N,2,L,L)
        self.S = torch.from_numpy(Sigma)[:, None, ...]  # (N,1,L,L)
        self.T = torch.from_numpy(T)[:, None]           # (N,1)
        self.P = torch.from_numpy(Phase)                # (N,)
        self.Y = torch.from_numpy(Y)[:, None, ...]      # (N,1,L,L)
        self.M = torch.from_numpy(Mask)[:, None, ...]   # (N,1,L,L)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return {
            "X": self.X[i],
            "S": self.S[i],
            "T": self.T[i],
            "P": self.P[i],
            "Y": self.Y[i],
            "M": self.M[i]
        }

full_ds = IsingCSVDataset(X, Sigma, T, Phase, Y, Mask)

# Split (80/10/10)
n_train = int(0.8 * N)
n_val   = int(0.1 * N)
n_test  = N - n_train - n_val
g = torch.Generator().manual_seed(SEED)
train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=g)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==========================
# Enhanced Device selection
# ==========================
def setup_device(device_preference="cuda"):
    """Setup device with GPU optimization if available"""
    if device_preference == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"?? Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("?? Using CPU (GPU not available or not preferred)")
    return device

device = setup_device(DEVICE)

# ==========================
# Enhanced Physics helpers with noise-aware functions
# ==========================
class PeriodicConv(nn.Module):
    """3x3 conv with circular padding via F.pad; kernel is a non-trainable buffer."""
    def __init__(self, kernel_3x3):
        super().__init__()
        k = torch.tensor(kernel_3x3, dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("weight", k)
    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="circular")
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=0)

_NN_K = [[0,1,0],[1,0,1],[0,1,0]]
_nn_conv = PeriodicConv(_NN_K).to(device)

def energy_per_site(s):  # s: (N,1,L,L)
    """Calculate energy per site more robustly"""
    nbr = _nn_conv(s)
    s_clamped = torch.sign(s)  # Force to ±1
    energy = - (s_clamped * nbr)
    return energy.mean(dim=(1,2,3)) / 2.0  # Divide by 2 to avoid double-counting

def nn_corr(s):
    nbr = _nn_conv(s)
    s_clamped = torch.sign(s)
    return (s_clamped * nbr).mean(dim=(1,2,3)) / 4.0

def c2_corr(s):
    """
    Longer-range correlation C2 at a separation of two lattice spacings
    along the horizontal and vertical lattice directions, with periodic
    boundary conditions.

    C1 is retained for evaluation only; C2 is used as the independent
    correlation regularizer during training.
    """
    s_clamped = torch.sign(s)
    nbr2 = (
        torch.roll(s, shifts=2, dims=2)
        + torch.roll(s, shifts=-2, dims=2)
        + torch.roll(s, shifts=2, dims=3)
        + torch.roll(s, shifts=-2, dims=3)
    )
    return (s_clamped * nbr2).mean(dim=(1,2,3)) / 4.0

def mag_abs(s):
    s_clamped = torch.sign(s)
    return s_clamped.mean(dim=(2,3)).abs().squeeze(1)

def binary_consistency_loss(pred, threshold=0.8):
    """Encourage predictions to be near -1 or +1"""
    return torch.mean((pred.abs() - 1.0).pow(2))

# ==========================
# ENHANCED Model
# ==========================
class EnhancedInvNet(nn.Module):
    def __init__(self, L):
        super().__init__()
        C = 64
        
        # Encoder with residual connections
        self.enc_conv1 = nn.Conv2d(2, C, 3, padding=1, padding_mode="circular")
        self.enc_conv2 = nn.Conv2d(C, C, 3, padding=1, padding_mode="circular")
        self.enc_conv3 = nn.Conv2d(C, C, 3, padding=1, padding_mode="circular")
        self.enc_conv4 = nn.Conv2d(C, C, 3, padding=1, padding_mode="circular")
        self.enc_conv5 = nn.Conv2d(C, C, 3, padding=1, padding_mode="circular")
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(C, C//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C//4, 1, 1),
            nn.Sigmoid()
        )
        
        # Decoder
        self.dec_conv1 = nn.Conv2d(C + 1, 64, 3, padding=1, padding_mode="circular")
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1, padding_mode="circular")
        self.dec_conv3 = nn.Conv2d(64, 32, 3, padding=1, padding_mode="circular")
        self.dec_conv4 = nn.Conv2d(32, 16, 3, padding=1, padding_mode="circular")
        self.dec_conv5 = nn.Conv2d(16, 1, 1)
        
        # Heads for temperature and phase
        self.head_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self.head_P = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        noisy_input = x[:, :1, :, :]
        mask = x[:, 1:, :, :]
        
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
        
        S_final = mask * noisy_input + (1 - mask) * S_hat
        
        T_hat = self.head_T(h5)
        P_log = self.head_P(h5)
        
        return S_final, T_hat, P_log, attention_map

model_init_start = time.time()
net = EnhancedInvNet(L).to(device)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
ce = nn.CrossEntropyLoss()
model_init_time = time.time() - model_init_start

print(f"? Model initialized in {model_init_time:.2f}s")
print(f"?? Model parameters: {sum(p.numel() for p in net.parameters()):,}")

# ==========================
# Training
# ==========================
train_hist, val_hist = [], []
train_rec_hist, val_rec_hist = [], []
best_val_loss = float('inf')
patience_counter = 0
training_start = time.time()

print("\n?? Starting enhanced training...")
for epoch in range(1, EPOCHS+1):
    epoch_start = time.time()
    net.train()
    epoch_loss = 0.0
    epoch_rec_loss = 0.0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS} [Train]",
                      leave=False, ncols=100)
    
    for batch_idx, batch in enumerate(train_pbar):
        Xb = batch["X"].to(device, non_blocking=True)
        S = batch["S"].to(device, non_blocking=True)
        Tt = batch["T"].to(device, non_blocking=True)
        P = batch["P"].to(device, non_blocking=True)
        Yb = batch["Y"].to(device, non_blocking=True)
        M = batch["M"].to(device, non_blocking=True)

        S_hat, T_hat, P_log, attention = net(Xb)

        observed_rec = (M * (S_hat - Yb).abs()).sum() / (M.sum() + 1e-8)
        missing_rec = ((1 - M) * (S_hat - S).abs()).sum() / ((1 - M).sum() + 1e-8)
        rec = observed_rec + 2.0 * missing_rec
        
        E_hat = energy_per_site(S_hat).mean()
        E_tgt = energy_per_site(S).mean().detach()
        L_E = (E_hat - E_tgt).pow(2)

        M_hat = mag_abs(S_hat).mean()
        M_tgt = mag_abs(S).mean().detach()
        L_M = (M_hat - M_tgt).pow(2)

        C2_hat = c2_corr(S_hat).mean()
        C2_tgt = c2_corr(S).mean().detach()
        L_C2 = (C2_hat - C2_tgt).pow(2)
        
        L_bin = binary_consistency_loss(S_hat)

        L_T = (T_hat - Tt).abs().mean()
        L_Ph = ce(P_log, P)

        loss = (LAM_REC * rec + L_E * LAM_E + L_M * LAM_M +
                L_C2 * LAM_C2 + L_T * LAM_T + L_Ph * LAM_PHASE +
                L_bin * LAM_BIN)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        
        epoch_loss += float(loss.item())
        epoch_rec_loss += float(rec.item())
        
        if batch_idx % 10 == 0:
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rec': f'{rec.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

    epoch_loss /= max(1, len(train_loader))
    epoch_rec_loss /= max(1, len(train_loader))
    train_hist.append(epoch_loss)
    train_rec_hist.append(epoch_rec_loss)

    # Validation
    net.eval()
    val_loss = 0.0
    val_rec_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            Xb = batch["X"].to(device)
            S = batch["S"].to(device)
            Tt = batch["T"].to(device)
            P = batch["P"].to(device)
            Yb = batch["Y"].to(device)
            M = batch["M"].to(device)

            S_hat, T_hat, P_log, _ = net(Xb)
            
            observed_rec = (M * (S_hat - Yb).abs()).sum() / (M.sum() + 1e-8)
            missing_rec = ((1 - M) * (S_hat - S).abs()).sum() / ((1 - M).sum() + 1e-8)
            rec = observed_rec + 2.0 * missing_rec
            
            L_E = (energy_per_site(S_hat).mean() - energy_per_site(S).mean()).pow(2)
            L_M = (mag_abs(S_hat).mean() - mag_abs(S).mean()).pow(2)
            L_C2 = (c2_corr(S_hat).mean() - c2_corr(S).mean()).pow(2)
            L_bin = binary_consistency_loss(S_hat)
            L_T = (T_hat - Tt).abs().mean()
            L_Ph = ce(P_log, P)

            batch_val_loss = float((LAM_REC * rec + L_E * LAM_E + L_M * LAM_M +
                                    L_C2 * LAM_C2 + L_T * LAM_T + L_Ph * LAM_PHASE +
                                    L_bin * LAM_BIN).item())
            val_loss += batch_val_loss
            val_rec_loss += float(rec.item())
    
    val_loss /= max(1, len(val_loader))
    val_rec_loss /= max(1, len(val_loader))
    val_hist.append(val_loss)
    
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
    epoch_time = time.time() - epoch_start
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'val_loss': best_val_loss,
            'val_rec_loss': val_rec_loss,
        }, os.path.join(OUT_DIR, "best_model.pth"))
        improvement_flag = "??"
    else:
        patience_counter += 1
        improvement_flag = ""

    print(f"{improvement_flag} Epoch {epoch:03d} | "
          f"Train: {epoch_loss:.4f} (rec: {epoch_rec_loss:.4f}) | "
          f"Val: {val_loss:.4f} (rec: {val_rec_loss:.4f}) | "
          f"Time: {epoch_time:.1f}s | LR: {current_lr:.2e} | "
          f"Patience: {patience_counter}/{PATIENCE}")
    
    if patience_counter >= PATIENCE:
        print(f"?? Early stopping triggered at epoch {epoch}")
        break

training_time = time.time() - training_start
print(f"? Training completed in {training_time:.2f}s ({training_time/60:.1f} min)")

# Load best model for evaluation
if os.path.exists(os.path.join(OUT_DIR, "best_model.pth")):
    checkpoint = torch.load(os.path.join(OUT_DIR, "best_model.pth"), map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    print(f"? Loaded best model from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.4f}")
else:
    print("?? No best model found, using final model for evaluation")
    checkpoint = {"val_rec_loss": 0.0}

# ==========================
# Test-time Evaluation
# ==========================
eval_start = time.time()
net.eval()
all_true_T, all_pred_T, all_true_P, all_pred_P = [], [], [], []
all_true_E, all_pred_E = [], []
all_true_M, all_pred_M = [], []
all_true_C1, all_pred_C1 = [], []
all_true_C2, all_pred_C2 = [], []
all_pred_S_sign, all_true_S, all_noisy_Y, all_mask = [], [], [], []
all_attention_maps = []

print("?? Running enhanced evaluation...")
with torch.no_grad():
    eval_pbar = tqdm(test_loader, desc="Evaluating", ncols=100)
    for batch in eval_pbar:
        Xb = batch["X"].to(device)
        S = batch["S"].to(device)
        Tt = batch["T"].to(device)
        P = batch["P"].to(device)
        Yb = batch["Y"].to(device)
        M = batch["M"].to(device)

        S_hat, T_hat, P_log, attention_map = net(Xb)
        S_sign = torch.where(S_hat > 0, 1.0, -1.0).float()

        all_true_T.append(Tt.cpu().numpy())
        all_pred_T.append(T_hat.cpu().numpy())
        all_true_P.append(P.cpu().numpy())
        all_pred_P.append(P_log.argmax(1).cpu().numpy())

        all_true_E.append(energy_per_site(S).cpu().numpy())
        all_pred_E.append(energy_per_site(S_sign).cpu().numpy())

        all_true_M.append(mag_abs(S).cpu().numpy())
        all_pred_M.append(mag_abs(S_sign).cpu().numpy())

        all_true_C1.append(nn_corr(S).cpu().numpy())
        all_pred_C1.append(nn_corr(S_sign).cpu().numpy())

        all_true_C2.append(c2_corr(S).cpu().numpy())
        all_pred_C2.append(c2_corr(S_sign).cpu().numpy())

        all_pred_S_sign.append(S_sign.cpu().numpy())
        all_true_S.append(S.cpu().numpy())
        all_noisy_Y.append(Yb.cpu().numpy())
        all_mask.append(M.cpu().numpy())
        all_attention_maps.append(attention_map.cpu().numpy())

T_true  = np.concatenate(all_true_T, axis=0).reshape(-1)
T_pred  = np.concatenate(all_pred_T, axis=0).reshape(-1)
P_true  = np.concatenate(all_true_P, axis=0).reshape(-1)
P_pred  = np.concatenate(all_pred_P, axis=0).reshape(-1)

E_true  = np.concatenate(all_true_E, axis=0).reshape(-1)
E_pred  = np.concatenate(all_pred_E, axis=0).reshape(-1)
M_true  = np.concatenate(all_true_M, axis=0).reshape(-1)
M_pred  = np.concatenate(all_pred_M, axis=0).reshape(-1)
C_true  = np.concatenate(all_true_C1, axis=0).reshape(-1)
C_pred  = np.concatenate(all_pred_C1, axis=0).reshape(-1)
C2_true = np.concatenate(all_true_C2, axis=0).reshape(-1)
C2_pred = np.concatenate(all_pred_C2, axis=0).reshape(-1)

S_pred  = np.concatenate(all_pred_S_sign, axis=0)
S_true  = np.concatenate(all_true_S, axis=0)
Y_test  = np.concatenate(all_noisy_Y, axis=0)
M_test  = np.concatenate(all_mask, axis=0)
att_maps = np.concatenate(all_attention_maps, axis=0)

eval_time = time.time() - eval_start

mae_T   = float(np.mean(np.abs(T_true - T_pred)))
acc_P   = float(np.mean(P_true == P_pred))
miss    = (M_test == 0)
imp_acc = float(((S_pred == S_true) & miss).sum() / max(1, miss.sum()))

obs_acc = float(((S_pred == S_true) & (M_test == 1)).sum() / max(1, (M_test == 1).sum()))
overall_acc = float((S_pred == S_true).sum() / max(1, S_pred.size))

mae_E  = float(np.mean(np.abs(E_true - E_pred)))
mae_M  = float(np.mean(np.abs(M_true - M_pred)))
mae_C1 = float(np.mean(np.abs(C_true - C_pred)))
mae_C2 = float(np.mean(np.abs(C2_true - C2_pred)))

pd.DataFrame([{
    "Temperature_MAE": mae_T,
    "Phase_Accuracy": acc_P,
    "Imputation_Accuracy_on_missing": imp_acc,
    "Observation_Accuracy": obs_acc,
    "Overall_Accuracy": overall_acc,
    "Energy_MAE": mae_E,
    "Magnetization_MAE": mae_M,
    "C1_MAE": mae_C1,
    "C2_MAE": mae_C2,
    "N_test": int(T_true.shape[0]),
    "Final_Val_Loss": float(best_val_loss),
    "Final_Val_Rec_Loss": float(checkpoint.get('val_rec_loss', 0)),
}]).to_csv(os.path.join(OUT_DIR, "metrics_global.csv"), index=False)

print("\n=== Enhanced Test Results ===")
print(pd.read_csv(os.path.join(OUT_DIR, "metrics_global.csv")).to_string(index=False))

pd.DataFrame({
    "T_true": T_true, "T_pred": T_pred,
    "Phase_true": P_true, "Phase_pred": P_pred,
    "E_true": E_true, "E_pred": E_pred,
    "|M|_true": M_true, "|M|_pred": M_pred,
    "C1_true": C_true, "C1_pred": C_pred,
    "C2_true": C2_true, "C2_pred": C2_pred,
}).to_csv(os.path.join(OUT_DIR, "preds_test.csv"), index=False)

# ==========================
# Per-temperature stats with proper Heat capacity (ENHANCED)
# ==========================

def calculate_heat_capacity_from_derivative(temps, energies):
    """
    Calculate heat capacity as Cv = d<E>/dT using smoothed finite differences.
    More robust than fluctuation method for reconstructed data.
    """
    if len(temps) < 2:
        return np.zeros_like(temps)
    
    # Sort by temperature
    sorted_idx = np.argsort(temps)
    temps_sorted = temps[sorted_idx]
    energies_sorted = energies[sorted_idx]
    
    # Apply smoothing to reduce noise
    from scipy.ndimage import gaussian_filter1d
    if len(temps_sorted) > 5:
        energies_smoothed = gaussian_filter1d(energies_sorted, sigma=1.0)
    else:
        energies_smoothed = energies_sorted
    
    cv = np.zeros_like(temps_sorted)
    
    # Use central difference for interior points
    for i in range(1, len(temps_sorted)-1):
        dE = energies_smoothed[i+1] - energies_smoothed[i-1]
        dT = temps_sorted[i+1] - temps_sorted[i-1]
        if abs(dT) > 1e-8:  # Avoid division by zero
            cv[i] = dE / dT
    
    # Handle boundaries with forward/backward difference
    if len(temps_sorted) >= 2:
        # First point: forward difference
        dE = energies_smoothed[1] - energies_smoothed[0]
        dT = temps_sorted[1] - temps_sorted[0]
        if abs(dT) > 1e-8:
            cv[0] = dE / dT
        
        # Last point: backward difference
        dE = energies_smoothed[-1] - energies_smoothed[-2]
        dT = temps_sorted[-1] - temps_sorted[-2]
        if abs(dT) > 1e-8:
            cv[-1] = dE / dT
    
    return cv

# Calculate heat capacity using both methods for comparison
unique_T = np.sort(np.unique(T_true))
T_centers = []
E_true_means, E_pred_means = [], []
M_true_means, M_pred_means = [], []
C1_true_means, C1_pred_means = [], []
C2_true_means, C2_pred_means = [], []

# First pass: collect means for each unique temperature
for T0 in unique_T:
    idx = np.isclose(T_true, T0, rtol=1e-05, atol=1e-08)
    if not np.any(idx) or idx.sum() < 3:  # Need at least 3 points for meaningful stats
        continue
    
    T_centers.append(float(T0))
    E_true_means.append(float(np.mean(E_true[idx])))
    E_pred_means.append(float(np.mean(E_pred[idx])))
    M_true_means.append(float(np.mean(M_true[idx])))
    M_pred_means.append(float(np.mean(M_pred[idx])))
    C1_true_means.append(float(np.mean(C_true[idx])))
    C1_pred_means.append(float(np.mean(C_pred[idx])))
    C2_true_means.append(float(np.mean(C2_true[idx])))
    C2_pred_means.append(float(np.mean(C2_pred[idx])))

# Convert to arrays
T_centers = np.array(T_centers)
E_true_means = np.array(E_true_means)
E_pred_means = np.array(E_pred_means)
M_true_means = np.array(M_true_means)
M_pred_means = np.array(M_pred_means)
C1_true_means = np.array(C1_true_means)
C1_pred_means = np.array(C1_pred_means)
C2_true_means = np.array(C2_true_means)
C2_pred_means = np.array(C2_pred_means)

# Calculate heat capacity using derivative method
Cv_true_deriv = calculate_heat_capacity_from_derivative(T_centers, E_true_means)
Cv_pred_deriv = calculate_heat_capacity_from_derivative(T_centers, E_pred_means)

# Also calculate using fluctuation method for reference
Cv_true_fluct = np.zeros_like(T_centers)
Cv_pred_fluct = np.zeros_like(T_centers)

for i, T0 in enumerate(T_centers):
    idx = np.isclose(T_true, T0, rtol=1e-05, atol=1e-08)
    if idx.sum() >= 10:  # Need enough samples for fluctuation method
        e_true_vals = E_true[idx]
        e_pred_vals = E_pred[idx]
        if T0 > 1e-8:
            Cv_true_fluct[i] = float(np.var(e_true_vals, ddof=1) / (T0**2))
            Cv_pred_fluct[i] = float(np.var(e_pred_vals, ddof=1) / (T0**2))

# Second pass: compile all statistics
rows = []
for i, T0 in enumerate(T_centers):
    idx = np.isclose(T_true, T0, rtol=1e-05, atol=1e-08)
    if not np.any(idx):
        continue
    
    count = int(idx.sum())
    e_true_vals = E_true[idx]
    e_pred_vals = E_pred[idx]
    m_true_vals = M_true[idx]
    m_pred_vals = M_pred[idx]
    c1_true_vals = C_true[idx]
    c1_pred_vals = C_pred[idx]
    c2_true_vals = C2_true[idx]
    c2_pred_vals = C2_pred[idx]
    
    # Accuracy calculations
    miss_T = miss[idx]
    S_pred_T = S_pred[idx]
    S_true_T = S_true[idx]
    
    bin_missing_acc = float(((S_pred_T == S_true_T) & miss_T).sum() /
                             max(1, miss_T.sum()))
    bin_obs_acc = float(((S_pred_T == S_true_T) & (~miss_T)).sum() /
                         max(1, (~miss_T).sum()))
    bin_overall_acc = float((S_pred_T == S_true_T).sum() /
                             max(1, S_pred_T.size))
    
    # Choose which Cv method to use (derivative is more robust for reconstruction)
    use_derivative_method = True  # Set to False to use fluctuation method
    
    if use_derivative_method:
        cv_true_val = Cv_true_deriv[i] if i < len(Cv_true_deriv) else 0.0
        cv_pred_val = Cv_pred_deriv[i] if i < len(Cv_pred_deriv) else 0.0
    else:
        cv_true_val = Cv_true_fluct[i] if i < len(Cv_true_fluct) else 0.0
        cv_pred_val = Cv_pred_fluct[i] if i < len(Cv_pred_fluct) else 0.0
    
    # Clip negative Cv values (unphysical)
    cv_true_val = max(0.0, cv_true_val)
    cv_pred_val = max(0.0, cv_pred_val)
    
    rows.append({
        "T_center": float(T0),
        "count": count,
        "E_true": float(np.mean(e_true_vals)),
        "E_pred": float(np.mean(e_pred_vals)),
        "Mabs_true": float(np.mean(m_true_vals)),
        "Mabs_pred": float(np.mean(m_pred_vals)),
        "C1_true": float(np.mean(c1_true_vals)),
        "C1_pred": float(np.mean(c1_pred_vals)),
        "C2_true": float(np.mean(c2_true_vals)),
        "C2_pred": float(np.mean(c2_pred_vals)),
        "Cv_true": cv_true_val,
        "Cv_pred": cv_pred_val,
        "Cv_true_fluct": float(Cv_true_fluct[i]) if i < len(Cv_true_fluct) else 0.0,
        "Cv_pred_fluct": float(Cv_pred_fluct[i]) if i < len(Cv_pred_fluct) else 0.0,
        "Accuracy_missing": bin_missing_acc,
        "Accuracy_observed": bin_obs_acc,
        "Accuracy_overall": bin_overall_acc,
    })

perbin_df = pd.DataFrame(rows).sort_values("T_center").reset_index(drop=True)
perbin_df.to_csv(os.path.join(OUT_DIR, "metrics_per_temp_bin.csv"), index=False)

# Optional: Also save a version with both Cv methods for comparison
perbin_df_comparison = perbin_df.copy()
perbin_df_comparison["Cv_method"] = "derivative" if use_derivative_method else "fluctuation"
perbin_df_comparison.to_csv(os.path.join(OUT_DIR, "metrics_per_temp_bin_with_Cv_comparison.csv"), index=False)

print(f"?? Calculated heat capacity using {'derivative' if use_derivative_method else 'fluctuation'} method")
print(f"?? Temperature points: {len(perbin_df)}")

# =========================================================
# Plot style
# =========================================================
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2.2,
    "figure.titlesize": 18,
    "figure.titleweight": "bold",
})

# =========================================================
# PDF 1: PHYSICS
# =========================================================
plot_start = time.time()
physics_pdf = os.path.join(OUT_DIR, "physics_plots.pdf")
with PdfPages(physics_pdf) as pdf:
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # |M|(T)
    axs[0,0].plot(perbin_df["T_center"], perbin_df["Mabs_true"], marker="o", color="black", label="True", linewidth=2)
    axs[0,0].plot(perbin_df["T_center"], perbin_df["Mabs_pred"], marker="s", color="dimgray", label="Recon", linewidth=2)
    axs[0,0].axvline(Tc, linestyle="--", color="red", alpha=0.8, label="Tc")
    axs[0,0].set_title("Magnetization |M| vs T")
    axs[0,0].set_xlabel("Temperature T")
    axs[0,0].set_ylabel("|M|")
    axs[0,0].legend()
    axs[0,0].grid(True, alpha=0.3)

    # E(T)
    axs[0,1].plot(perbin_df["T_center"], perbin_df["E_true"], marker="o", color="black", label="True", linewidth=2)
    axs[0,1].plot(perbin_df["T_center"], perbin_df["E_pred"], marker="s", color="dimgray", label="Recon", linewidth=2)
    axs[0,1].axvline(Tc, linestyle="--", color="red", alpha=0.8)
    axs[0,1].set_title("Energy per site vs T")
    axs[0,1].set_xlabel("Temperature T")
    axs[0,1].set_ylabel("Energy")
    axs[0,1].legend()
    axs[0,1].grid(True, alpha=0.3)

    # C1(T)
    axs[1,0].plot(perbin_df["T_center"], perbin_df["C1_true"], marker="o", color="black", label="True", linewidth=2)
    axs[1,0].plot(perbin_df["T_center"], perbin_df["C1_pred"], marker="s", color="dimgray", label="Recon", linewidth=2)
    axs[1,0].axvline(Tc, linestyle="--", color="red", alpha=0.8)
    axs[1,0].set_title("Nearest-neighbor correlation C1 vs T")
    axs[1,0].set_xlabel("Temperature T")
    axs[1,0].set_ylabel("C1")
    axs[1,0].legend()
    axs[1,0].grid(True, alpha=0.3)

    # Cv(T)
    axs[1,1].plot(perbin_df["T_center"], perbin_df["Cv_true"], marker="o", color="black", label="True", linewidth=2)
    axs[1,1].plot(perbin_df["T_center"], perbin_df["Cv_pred"], marker="s", color="dimgray", label="Recon", linewidth=2)
    axs[1,1].axvline(Tc, linestyle="--", color="red", alpha=0.8, label="Tc")
    axs[1,1].set_title("Heat capacity $C_v$ vs T")
    axs[1,1].set_xlabel("Temperature T")
    axs[1,1].set_ylabel("$C_v$")
    axs[1,1].legend()
    axs[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)

    # Page 2: Energy comparison
    with torch.no_grad():
        E_noisy = energy_per_site(torch.from_numpy(Y_test).to(device)).cpu().numpy().reshape(-1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Energy Analysis", fontweight='bold')
    
    ax1.scatter(E_true, E_pred, s=8, alpha=0.6, label="Reconstructed", color="black")
    ax1.scatter(E_true, E_noisy, s=8, alpha=0.4, label="Noisy input", color="red")
    mn = float(min(E_true.min(), E_pred.min(), E_noisy.min()))
    mx = float(max(E_true.max(), E_pred.max(), E_noisy.max()))
    ax1.plot([mn, mx], [mn, mx], linestyle="--", color="gray", linewidth=2, label="Perfect")
    ax1.set_xlabel("True Energy")
    ax1.set_ylabel("Predicted / Noisy Energy")
    ax1.set_title("Energy Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    energy_error_recon = np.abs(E_pred - E_true)
    energy_error_noisy = np.abs(E_noisy - E_true)
    ax2.hist(energy_error_recon, bins=50, alpha=0.7, label="Reconstruction", color="black", density=True)
    ax2.hist(energy_error_noisy, bins=50, alpha=0.7, label="Noisy input", color="red", density=True)
    ax2.set_xlabel("Absolute Energy Error")
    ax2.set_ylabel("Density")
    ax2.set_title("Energy Error Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)


    # Page 3: C2(T) longer-range correlation
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle("Longer-range Correlation Analysis", fontweight='bold')
    ax.plot(perbin_df["T_center"], perbin_df["C2_true"],
            marker="o", color="black", label="True", linewidth=2)
    ax.plot(perbin_df["T_center"], perbin_df["C2_pred"],
            marker="s", color="dimgray", label="Recon", linewidth=2)
    ax.axvline(Tc, linestyle="--", color="red", alpha=0.8, label="Tc")
    ax.set_title("Correlation $C_2$ at separation two vs T")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("$C_2$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)

# =========================================================
# PDF 2: TRAINING
# =========================================================
training_pdf = os.path.join(OUT_DIR, "training_plots.pdf")
with PdfPages(training_pdf) as pdf:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Progress", fontweight='bold')
    
    ax1.plot(train_hist, label="Train", color="black", linewidth=2)
    ax1.plot(val_hist, label="Validation", color="red", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.set_title("Total Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_rec_hist, label="Train Reconstruction", color="darkblue", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Reconstruction Loss")
    ax2.set_title("Reconstruction Loss Focus")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)

# =========================================================
# PDF 3: CONFIGURATIONS with attention maps
# =========================================================
configs_pdf = os.path.join(OUT_DIR, "configs_bw.pdf")
with PdfPages(configs_pdf) as pdf:
    def nearest_idx(targetT):
        return int(np.argmin(np.abs(T_true - targetT)))
    targets = [1.0, 2.0, Tc, 4.0]

    fig, axes = plt.subplots(len(targets), 4, figsize=(12, 3*len(targets)))
    #fig.suptitle("Enhanced Reconstruction with Attention Maps", fontweight='bold', y=0.95)
    
    if len(targets) == 1:
        axes = np.array([axes])

    for r, t0 in enumerate(targets):
        idx = nearest_idx(t0)
        s_clean = S_true[idx,0]
        y_obs  = Y_test[idx,0]
        s_rec  = S_pred[idx,0]
        att_map = att_maps[idx,0]
        
        ax = axes[r, 0]
        ax.imshow(s_clean, vmin=-1, vmax=1, cmap="gray", interpolation="nearest")
        ax.set_title(f"Clean (T={T_true[idx]:.2f})", fontweight="bold")
        ax.axis("off")
        
        ax = axes[r, 1]
        ax.imshow(y_obs, vmin=-1, vmax=1, cmap="gray", interpolation="nearest")
        ax.set_title("Noisy Input", fontweight="bold")
        ax.axis("off")
        
        ax = axes[r, 2]
        ax.imshow(att_map, cmap="hot", interpolation="nearest")
        ax.set_title("Attention Map", fontweight="bold")
        ax.axis("off")
        
        ax = axes[r, 3]
        ax.imshow(s_rec, vmin=-1, vmax=1, cmap="gray", interpolation="nearest")
        accuracy = float((s_rec == s_clean).mean())
        #ax.set_title(f"Reconstructed (Acc: {accuracy:.3f})", fontweight="bold")
        ax.set_title(f"Reconstructed", fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)

# =========================================================
# PDF 4: ACCURACY VS TEMPERATURE
# =========================================================
accuracy_pdf = os.path.join(OUT_DIR, "accuracy_vs_temperature.pdf")
with PdfPages(accuracy_pdf) as pdf:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(perbin_df["T_center"], perbin_df["Accuracy_missing"], marker="^", color="darkred", 
            label="Missing spins", linewidth=3, markersize=8)
    ax.plot(perbin_df["T_center"], perbin_df["Accuracy_observed"], marker="v", color="darkblue", 
            label="Observed spins", linewidth=3, markersize=8)
    ax.plot(perbin_df["T_center"], perbin_df["Accuracy_overall"], marker="o", color="black", 
            label="Overall", linewidth=3, markersize=8)
    
    ax.axvline(Tc, linestyle="--", color="red", alpha=0.8, linewidth=2, label="Tc")
    ax.set_xlabel("Temperature T", fontweight='bold', fontsize=14)
    ax.set_ylabel("Reconstruction Accuracy", fontweight='bold', fontsize=14)
    ax.set_title("Reconstruction Accuracy vs Temperature", fontweight='bold', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    avg_missing_acc = perbin_df["Accuracy_missing"].mean()
    avg_obs_acc = perbin_df["Accuracy_observed"].mean()
    avg_overall_acc = perbin_df["Accuracy_overall"].mean()
    
    textstr = '\n'.join([
        f'Average Missing: {avg_missing_acc:.3f}',
        f'Average Observed: {avg_obs_acc:.3f}',
        f'Average Overall: {avg_overall_acc:.3f}'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)

plot_time = time.time() - plot_start

# ==========================
# Timing information
# ==========================
total_time = time.time() - start_time

timing_df = pd.DataFrame([{
    "timestamp": timestamp,
    "total_time_seconds": total_time,
    "total_time_minutes": total_time/60,
    "data_loading_time": data_load_time,
    "model_init_time": model_init_time,
    "training_time": training_time,
    "training_time_minutes": training_time/60,
    "evaluation_time": eval_time,
    "plotting_time": plot_time,
    "epochs_completed": len(train_hist),
    "final_val_loss": best_val_loss,
    "final_val_rec_loss": checkpoint.get('val_rec_loss', 0),
    "early_stopping_triggered": patience_counter >= PATIENCE,
    "final_learning_rate": scheduler.get_last_lr()[0],
}])

timing_df.to_csv(os.path.join(OUT_DIR, "training_times.csv"), index=False)

# ==========================
# Final Summary
# ==========================
print("\n" + "="*70)
print("?? ENHANCED RUN COMPLETED SUCCESSFULLY!")
print("="*70)

timing_breakdown = {
    "Total Time": f"{total_time:.2f}s ({total_time/60:.1f} min)",
    "Data Loading": f"{data_load_time:.2f}s",
    "Model Init": f"{model_init_time:.2f}s", 
    "Training": f"{training_time:.2f}s ({training_time/60:.1f} min)",
    "Evaluation": f"{eval_time:.2f}s",
    "Plotting": f"{plot_time:.2f}s"
}

print("\n??  TIMING BREAKDOWN:")
for key, value in timing_breakdown.items():
    print(f"   {key:<15}: {value}")

print(f"\n?? PERFORMANCE SUMMARY:")
print(f"   Epochs completed: {len(train_hist)}/{EPOCHS}")
print(f"   Best validation loss: {best_val_loss:.4f}")
print(f"   Final training loss: {train_hist[-1]:.4f}")
print(f"   Temperature MAE: {mae_T:.4f}")
print(f"   Phase Accuracy: {acc_P:.3f}")
print(f"   Missing spin accuracy: {imp_acc:.3f}")
print(f"   Overall accuracy: {overall_acc:.3f}")
print(f"   Early stopping: {'Yes' if patience_counter >= PATIENCE else 'No'}")

if torch.cuda.is_available():
    print(f"   Device: GPU ({torch.cuda.get_device_name(0)})")
else:
    print(f"   Device: CPU")

print(f"\n?? OUTPUT FILES:")
files_to_check = [
    ("metrics_global.csv", "Global metrics"),
    ("metrics_per_temp_bin.csv", "Per-temperature metrics"), 
    ("preds_test.csv", "Predictions"),
    ("training_times.csv", "Timing data"),
    ("best_model.pth", "Best model weights"),
    ("physics_plots.pdf", "Physics plots"),
    ("training_plots.pdf", "Training curves"),
    ("configs_bw.pdf", "Configuration plots"),
    ("accuracy_vs_temperature.pdf", "Accuracy vs Temperature")
]

for filename, description in files_to_check:
    full_path = os.path.join(OUT_DIR, filename)
    if os.path.exists(full_path):
        file_size = os.path.getsize(full_path) / 1024  # KB
        print(f"   ? {description:<25} {filename:<28} ({file_size:.1f} KB)")
    else:
        print(f"   ? {description:<25} {filename:<28} (MISSING)")

print(f"\n?? ENHANCEMENTS APPLIED:")
enhancements = []
#enhancements = [
#    "Robust Phase mapping (F/P ? 1/0)",
#    "Robust spin column detection (Spin* / spin_*)",
#    "Advanced residual architecture with attention",
#    "Enhanced reconstruction loss (2× weight on missing spins)",
#    "Binary consistency loss for spin values",
#    "Gradient clipping and AdamW optimizer",
#    "Cosine annealing learning rate scheduler",
#    "Attention mechanism for noise regions",
#    "Per-temperature heat capacity without binning artefacts",
#    "Separate accuracy vs temperature PDF",
#    "Robust spin binarization for physics calculations"
#]

for enhancement in enhancements:
    print(f"   ? {enhancement}")

print(f"\n?? Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
