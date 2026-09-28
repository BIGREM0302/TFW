import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy.signal import stft
from scipy.stats import skew, kurtosis

# =========================
# Basic utils
# =========================

FS = 200          # sampling rate
WIN_SEC = 50
WIN_SAMPLES = FS * WIN_SEC

SAVE_DIR = "./figures"
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_eeg(df, title="", max_ch=20):
    fig, axs = plt.subplots(max_ch, 1, figsize=(25, 15), sharex=True)

    for i, ax in enumerate(axs):
        ax.plot(df.iloc[:, i], color="black", linewidth=0.6)
        ax.set_ylabel(df.columns[i], rotation=0, labelpad=25)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[:].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{title.replace(' ', '_')}.png", dpi=200, bbox_inches="tight")
    plt.close()

# =========================
# Wavelet denoising
# =========================

def maddest(d):
    return np.mean(np.abs(d - np.mean(d)))

def visualize_denoise(df, wavelet="db8", level=1, mode="per"):
    for ch in df.columns:
        signal = df[ch].values
        # 小波分解
        coeffs = pywt.wavedec(signal, wavelet, mode=mode)
        sigma = (1 / 0.6745) * maddest(coeffs[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(df)))
        
        # Threshold 前的 coeffs
        coeffs_before = coeffs.copy()
        # Threshold 後
        coeffs_thresholded = [
            pywt.threshold(c, uthresh, mode='hard') if i>0 else c
            for i, c in enumerate(coeffs)
        ]
        reconstructed = pywt.waverec(coeffs_thresholded, wavelet, mode=mode)[:len(signal)]

        n_levels = len(coeffs)
        fig, axs = plt.subplots(n_levels + 1, 1, figsize=(20, 2*(n_levels+1)))
        fig.suptitle(f"Channel: {ch}", fontsize=16)

        for i in range(n_levels):
            axs[i].plot(coeffs_before[i], color='gray', alpha=0.5, label='original coeff')
            if i > 0:
                axs[i].plot(coeffs_thresholded[i], color='red', alpha=0.8, label='thresholded coeff')
            axs[i].set_ylabel(f"c{i}")
            axs[i].legend(loc='upper right')

        axs[-1].plot(signal, color='gray', alpha=0.5, label='original signal')
        axs[-1].plot(reconstructed, color='blue', alpha=0.8, label='denoised signal')
        axs[-1].set_ylabel("signal")
        axs[-1].legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/denoise_{ch}.png", dpi=200, bbox_inches="tight")
        plt.show()

def denoise(df, wavelet="db8", level=1, mode="per"):
    out = {}

    for ch in df.columns:
        print(f'Processing channel {ch}')
        coeffs = pywt.wavedec(df[ch], wavelet, mode=mode)
        sigma = (1 / 0.6745) * maddest(coeffs[-level])
        print("len", len(df))
        uthresh = sigma * np.sqrt(2 * np.log(len(df)))

        coeffs[1:] = [
            pywt.threshold(c, uthresh, mode="hard")
            for c in coeffs[1:]
        ]

        out[ch] = pywt.waverec(coeffs, wavelet, mode=mode)[:len(df)]

    return pd.DataFrame(out)


# =========================
# Feature functions
# =========================

def energy(x):
    return np.sum(x ** 2)


def shannon_entropy(p):
    p = p / (np.sum(p) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))


def basic_stats(x):
    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "skew": skew(x),
        "kurtosis": kurtosis(x),
        "rms": np.sqrt(np.mean(x ** 2))
    }


# =========================
# WPD features
# =========================

def wpd_features(x, wavelet="db4", level=4, subband_ratio=(0.5,1.0)):
    wp = pywt.WaveletPacket(x, wavelet, mode="per", maxlevel=level)
    nodes = wp.get_level(level, order="freq") #order according to frequency
    energies = np.array([energy(n.data) for n in nodes])
    rel_energy = energies / (energies.sum() + 1e-12)
    features = {
        "WPD_total_energy": energies.sum(),
        "WPD_shannon_entropy": shannon_entropy(rel_energy),
        "WPD_log_energy_entropy": np.sum(np.log(energies + 1e-12)),
    }
    # RSWE 
    for i, re in enumerate(rel_energy):
        features[f"WPD_p_{i}"] = re 
    # -----------------------------
    # RSWE entropy (only for middel/high frequency sub-band)
    # -----------------------------
    n_nodes = len(nodes)
    start_idx = int(n_nodes * subband_ratio[0])
    end_idx = int(n_nodes * subband_ratio[1])
    selected_rel_energy = rel_energy[start_idx:end_idx]
    selected_rel_energy /= selected_rel_energy.sum() + 1e-12  # normalize
    RSWE_entropy = -np.sum(selected_rel_energy * np.log(selected_rel_energy + 1e-12))
    features["WPD_RSWE_entropy_mid_high"] = RSWE_entropy
    
    return features, energies


# =========================
# DWT + STFT features
# =========================

def plot_dwt_spectrum(x, wavelet="db4", level=5, FS=200):
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="per")
    
    plt.figure(figsize=(14, 2.5 * (level + 1)))

    # Approximation
    A = coeffs[0]
    N = len(A)
    fft_A = np.abs(np.fft.rfft(A))**2
    f = np.linspace(0, FS / (2**(level+1)), len(fft_A))

    plt.subplot(level + 1, 1, 1)
    plt.plot(f, fft_A)
    plt.title(f"A{level}: 0–{FS/(2**(level+1)):.2f} Hz")
    plt.ylabel("Power")

    # Details
    for i, D in enumerate(coeffs[1:], start=1):
        band_low = FS / (2**(level - i + 2))
        band_high = FS / (2**(level - i + 1))

        fft_D = np.abs(np.fft.rfft(D))**2
        f = np.linspace(band_low, band_high, len(fft_D))

        plt.subplot(level + 1, 1, i + 1)
        plt.plot(f, fft_D)
        plt.title(f"D{level - i + 1}: {band_low:.2f}–{band_high:.2f} Hz")
        plt.ylabel("Power")

    plt.xlabel("Frequency [Hz]")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/dwt_fft.png", dpi=200, bbox_inches="tight")
    plt.show()

# =========================
# Main (single case)
# =========================

if __name__ == "__main__":

    # ---- Load label row
    df_label = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    row = df_label.sample(n=1, random_state=60).iloc[0]
    print("Selected row:\n", row, "\n")

    # ---- Load EEG
    eeg = pd.read_parquet(
        f"/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/{row['eeg_id']}.parquet"
    )
    sp = pd.read_parquet(
        f"/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/{row['spectrogram_id']}.parquet"   
    )

    start = int(row["eeg_label_offset_seconds"]) * FS
    eeg = eeg.iloc[start:start + WIN_SAMPLES]


    print("EEG shape:", eeg.shape)
    print("Spectrogram shape:", sp.shape)
    # ---- Plot raw
    plot_eeg(eeg, title="Raw EEG")

    # ---- Denoise
    visualize_denoise(eeg) # For visualization
    eeg_denoised = denoise(eeg, wavelet="db8")
    plot_eeg(eeg_denoised, title="Denoised EEG")

    eeg_denoised = eeg
    # ---- Single channel analysis
    #eeg_denoised = eeg
    ch_name = eeg_denoised.columns[5]
    signal = eeg_denoised[ch_name].values

    print(f"\nUsing channel: {ch_name}")

    # ---- WPD
    wpd_feat, wpd_energy = wpd_features(signal)
    print("\nWPD features:")
    for k, v in wpd_feat.items():
        print(k, ":", v)

    plt.figure(figsize=(8, 3))
    plt.bar(range(len(wpd_energy)), wpd_energy)
    plt.title("WPD Energy Distribution")
    plt.xlabel("Subband")
    plt.ylabel("Energy")
    plt.show()

    # ---- DWT + FFT
    plot_dwt_spectrum(signal)
    print("\nPipeline finished ✔")
