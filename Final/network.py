import os
import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score

# =========================
# Configuration
# =========================
FS = 200                           # Sampling rate (Hz)
WIN_SEC = 50                       # Window length in seconds
WIN_SAMPLES = FS * WIN_SEC         # Number of samples per window
EEG_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Utility functions
# =========================
def maddest(d):
    """Mean absolute deviation from mean."""
    return np.mean(np.abs(d - np.mean(d)))

def energy(x):
    """Signal energy."""
    return np.sum(x ** 2)

def shannon_entropy(p):
    """Shannon entropy of a distribution."""
    p = p / (np.sum(p) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))

def basic_stats(x):
    """Basic statistical features of a signal."""
    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "skew": skew(x),
        "kurtosis": kurtosis(x),
        "rms": np.sqrt(np.mean(x ** 2))
    }

# =========================
# Wavelet feature extraction
# =========================
def wpd_features(x, wavelet="db4", level=3, subband_ratio=(0.5, 1.0)):
    """Extract Wavelet Packet Decomposition features."""
    wp = pywt.WaveletPacket(x, wavelet, mode="per", maxlevel=level)
    nodes = wp.get_level(level, order="freq")
    energies = np.array([energy(n.data) for n in nodes])
    rel_energy = energies / (energies.sum() + 1e-12)

    features = {
        "WPD_total_energy": energies.sum(),
        "WPD_shannon_entropy": shannon_entropy(rel_energy),
        "WPD_log_energy_entropy": np.sum(np.log(energies + 1e-12)),
    }

    # Relative subband energy (RSWE)
    for i, re in enumerate(rel_energy):
        features[f"WPD_p_{i}"] = re
    for i, n in enumerate(nodes):
        features[f"WPD_sig_{i}"] = n.data

    # Mid-high frequency RSWE entropy
    n_nodes = len(nodes)
    start_idx = int(n_nodes * subband_ratio[0])
    end_idx = int(n_nodes * subband_ratio[1])
    selected_rel_energy = rel_energy[start_idx:end_idx]
    selected_rel_energy /= selected_rel_energy.sum() + 1e-12
    features["WPD_RSWE_entropy_mid_high"] = -np.sum(selected_rel_energy * np.log(selected_rel_energy + 1e-12))

    return features

def dwt_basic_features(x, wavelet="db4", level=5):
    """Extract basic DWT features."""
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="per")
    features = {}
    for i, coef in enumerate(coeffs):
        e = energy(coef)
        stats = basic_stats(coef)
        prefix = "A" if i == 0 else f"D{i}"
        features[f"{prefix}_coef"] = coef
        features[f"{prefix}_energy"] = e
        for k, v in stats.items():
            features[f"{prefix}_{k}"] = v
    return features

def preprocess_eeg(eeg):
    """Interpolate missing values and fill NaNs."""
    eeg = eeg.copy()
    eeg = eeg.interpolate(limit_direction="both")
    eeg = eeg.fillna(0)
    return eeg

def denoise(df, wavelet="db8", level=1, mode="per"):
    """Wavelet denoising for all channels."""
    out = {}
    for ch in df.columns:
        coeffs = pywt.wavedec(df[ch], wavelet, mode=mode)
        sigma = (1 / 0.6745) * maddest(coeffs[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(df)))
        coeffs[1:] = [pywt.threshold(c, uthresh, mode="hard") for c in coeffs[1:]]
        out[ch] = pywt.waverec(coeffs, wavelet, mode=mode)[:len(df)]
    return pd.DataFrame(out)

def get_scalar_feature_columns(X_df):
    """Get list of scalar feature columns (exclude arrays)."""
    scalar_cols = [col for col in X_df.columns if np.isscalar(X_df[col].iloc[0])]
    return scalar_cols

# =========================
# Visualization utilities
# =========================
def plot_feature_scatter(X, y, f1, f2, figsize=(6, 5)):
    df_plot = pd.DataFrame({f1: X[f1], f2: X[f2], "label": y.values})
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df_plot, x=f1, y=f2, hue="label", palette="tab10", alpha=0.7)
    plt.title(f"{f1} vs {f2}")
    plt.tight_layout()
    plt.show()

def plot_feature_vs_class(X, y, feature, figsize=(6, 4)):
    df_plot = pd.DataFrame({feature: X[feature], "label": y.values})
    plt.figure(figsize=figsize)
    sns.violinplot(data=df_plot, x="label", y=feature, inner="quartile")
    plt.title(f"{feature} vs Class")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# =========================
# Dataset preparation
# =========================
def prepare_dataset(df_label, eeg_dir=EEG_DIR, channels=None):
    """Load EEG signals, extract WPD and DWT features."""
    X, y = [], []

    for _, row in tqdm(df_label.iterrows(), total=len(df_label)):
        eeg_path = os.path.join(eeg_dir, f"{row['eeg_id']}.parquet")
        if not os.path.exists(eeg_path):
            continue

        eeg = pd.read_parquet(eeg_path)
        start = int(row['eeg_label_offset_seconds']) * FS
        eeg = eeg.iloc[start:start + WIN_SAMPLES]
        eeg = preprocess_eeg(eeg)
        eeg = denoise(eeg, wavelet="db8")

        channels_use = eeg.columns.tolist() if channels is None else [ch for ch in channels if ch in eeg.columns]

        feat = {}
        for ch in channels_use:
            sig = eeg[ch].values
            feat.update({f"{ch}_{k}": v for k, v in wpd_features(sig).items()})
            feat.update({f"{ch}_{k}": v for k, v in dwt_basic_features(sig).items()})

        X.append(feat)
        y.append(row['expert_consensus'])

    X_df = pd.DataFrame(X)
    y_series = pd.Series(y, name='label')
    return X_df, y_series

# =========================
# Build WPD tensor for deep learning
# =========================
def build_wpd_tensor(X_df, channels, level=3):
    """Convert extracted WPD features into (N, C_freq, C_channel, W_time) tensor."""
    n_samples = len(X_df)
    n_bands = 2 ** level
    n_channels = len(channels)
    W = len(X_df.iloc[0][f"{channels[0]}_WPD_sig_0"])
    X_tensor = np.zeros((n_samples, n_bands, n_channels, W), dtype=np.float32)

    for i in tqdm(range(n_samples)):
        for h, ch in enumerate(channels):
            for c in range(n_bands):
                key = f"{ch}_WPD_sig_{c}"
                X_tensor[i, c, h, :] = X_df.iloc[i][key]

    return X_tensor

# =========================
# PyTorch dataset & dataloader
# =========================
class EEGTensorDataset(Dataset):
    """PyTorch Dataset for EEG tensors."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y)
        if self.y.ndim == 2:
            self.y = self.y.float()
        else:
            self.y = self.y.long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =========================
# EEG classifier model
# =========================
class EEGClassifier(nn.Module):
    def __init__(self, n_freq, n_ch, n_classes, W_time):
        super().__init__()
        # Temporal convolution
        self.temporal = nn.Sequential(
            nn.Conv2d(n_freq, 32, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        # Spatial convolution
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(n_ch, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        # Pooling + dropout
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.5)
        )
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (W_time // 4), 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# =========================
# Training & checkpoint utilities
# =========================
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
    }
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    if optimizer:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler:
        scheduler.load_state_dict(state["scheduler_state"])
    start_epoch = state["epoch"] + 1
    return model, optimizer, scheduler, start_epoch

def run_epoch(model, loader, criterion, optimizer, train=True):
    """Train or evaluate one epoch."""
    model.train() if train else model.eval()
    total_loss, all_preds, all_targets = 0, [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc

if __name__ == "__main__":
    # =========================
    # Load labels and sample
    # =========================
    df_label = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")

    # Only use 500 samples for fast run
    df_label = df_label.sample(n=500, random_state=42)

    channels = ['F3', 'F4', 'C3', 'C4', 'Cz', 'Pz']  # Selected EEG channels

    # =========================
    # Prepare dataset
    # =========================
    print("==> Preparing dataset and extracting features ...")
    X_df, y_series = prepare_dataset(df_label, channels=channels)

    # Stratified train/validation split
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )

    print("Train size:", X_train_df.shape)
    print("Validation size:", X_val_df.shape)
    print("Class distribution (train):\n", y_train.value_counts())
    print("Class distribution (val):\n", y_val.value_counts())

    # =========================
    # Encode labels and one-hot if needed
    # =========================
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    classes = le.classes_
    n_classes = len(classes)
    print("Classes:", classes)

    # =========================
    # Build WPD tensor for CNN input
    # =========================
    print("==> Building WPD tensors ...")
    X_train = build_wpd_tensor(X_train_df, channels=channels, level=3)
    X_val   = build_wpd_tensor(X_val_df,   channels=channels, level=3)

    # =========================
    # Normalize data
    # =========================
    mean = X_train.mean(axis=(0,2,3), keepdims=True)
    std  = X_train.std(axis=(0,2,3), keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std

    # =========================
    # Create PyTorch datasets and dataloaders
    # =========================
    batch_size = 16
    num_workers = 2
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    y_train_idx = y_train.map(label2idx).values
    y_val_idx   = y_val.map(label2idx).values

    train_ds = EEGTensorDataset(X_train, y_train_idx)
    val_ds   = EEGTensorDataset(X_val, y_val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # =========================
    # Initialize model, loss, optimizer, scheduler
    # =========================
    model = EEGClassifier(
        n_freq=X_train.shape[1],
        n_ch=X_train.shape[2],
        n_classes=n_classes,
        W_time=X_train.shape[3]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # =========================
    # Training loop with checkpoint
    # =========================
    best_val_acc = 0.0
    epochs = 100

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, train=False)

        # Scheduler step
        scheduler.step(val_acc)

        print(f"[{epoch:02d}] Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Save checkpoint every epoch
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:02d}.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)

        # Save best model separately
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, best_model_path)
            print(f"  -> New best model saved at epoch {epoch} with val_acc {val_acc:.4f}")
