import numpy as np
from scipy.io.wavfile import write
from scipy.signal import stft
import matplotlib.pyplot as plt

def gwave(a, b, c, T, Fs, sign=1, filename="output.wav"):
    # time axis
    t = np.linspace(0, T, int(Fs * T), endpoint=False)
    
    # f
    f = sign * (a * t**2 + b * t + c)
    
    # integral of f
    phase = 2 * np.pi * (a * (t**3) / 3 + b * (t**2) / 2 + c * t)
    
    # cosine
    x = np.cos(phase)

    x_int16 = np.int16(x / np.max(np.abs(x)) * 32767)
    write(filename, Fs, x_int16)
    print(f"WAV file saved: {filename}")

    # -------------------
    # f(t)
    plt.figure(figsize=(8,4))
    plt.plot(t, f, color='blue')
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Instantaneous Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("frequency_vs_time.png")
    print("Frequency vs time plot saved: frequency_vs_time.png")
    plt.close()

    # -------------------
    # STFT
    f_stft, t_stft, Zxx = stft(x, Fs, nperseg=1024)
    magnitude = np.abs(Zxx)

    f_max = np.max(np.abs(f)) * 1.1 # 10% margin

    plt.figure(figsize=(8,4))
    plt.pcolormesh(t_stft, f_stft, magnitude, shading='gouraud')
    plt.ylim(0, f_max)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title("Spectrogram (STFT)")
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.savefig("spectrogram.png")
    print("Spectrogram plot saved: spectrogram.png")
    plt.close()


if __name__ == "__main__":
    print("Frequency: ±(at^2+bt+c) Hz, t in seconds")
    a = float(input("Enter a (default -45): ") or -45)
    b = float(input("Enter b (default 520): ") or 520)
    c = float(input("Enter c (default 20): ") or 20)
    T = float(input("Enter T (duration in seconds, default 10): ") or 10)
    Fs = int(input("Enter Fs (sampling rate, default 44100): ") or 44100)
    filename = input("Enter output filename (default output.wav): ") or "output.wav"
    
    gwave(a, b, c, T, Fs, filename=filename)
