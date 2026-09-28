import numpy as np
import matplotlib.pyplot as plt
import time

np.set_printoptions(threshold=10)

def recSTFT(dt, df, x, t, f, B, verbose = False):
    """
    Rectangular-window STFT using FFT method (with phase correction)
    Δt * Δf = 1/N is satisfied (dt=df=0.05 → N=400)
    """
    # ==== Parameter Setting ====
    N = round(1.0 / (dt * df))  # ensure Δt * Δf = 1/N
    Q = round(B / dt)         # half window in samples
    T = len(t)
    F = len(f)
    m = f/df
    n = t/dt
    #m = np.arange(-F/2, F/2)
    print(f"m1={m%N}")

    if verbose:
        print("═" * 50)
        print("╔══════════════════ Debug Info ══════════════════╗")
        print("║")
        print(f"║ dt: {dt}")
        print(f"║  t: {t}")
        print(f"║  n: {n}")
        print(f"║ df: {df}")
        print(f"║  f: {f}")
        print(f"║  m: {m}")
        print(f"║  N: {N}")
        print(f"║  B: {B}")
        print(f"║  Q: {Q}")
        print(f"║  T: {T}")
        print(f"║  F: {F}")
        print("║")

        if N >= 2 * Q + 1:
            print("║ ✅ N ≥ 2Q+1 → Satisfy constraint (ヾﾉ・ω・`)")
        else:
            print("║ ❌ N < 2Q+1 → Error ٩( ˵ᐛ ˵)۶")

        print("║")
        print("╚════════════════════════════════════════════════╝")
        print("═" * 50)


    Y = np.zeros((T, F), dtype=complex)
  
    for n in range(T):
        # construct rectangular windowed segment centered at n
        n1 = max(0, n - Q)
        n2 = min(T, n + Q)
        x1 = np.zeros(N, dtype=complex)
        seg = x[n1:n2]
        x1[:len(seg)] = seg
        
        # FFT
        X1 = np.fft.fft(x1, N)
        # phase correction
        phase_corr = np.exp(1j * 2 * np.pi * (Q - n) * m / N)
        m1 = m % N
        m1 = m1.astype(int)
        Y[n, :] = dt * phase_corr * X1[m1]

    return Y


# ===== main =====
if __name__ == "__main__":
    dt = 0.05 # Δt = 0.05
    df = 0.05 # Δf = 0.05
    t = np.arange(0, 30+dt, dt)  
    f = np.arange(-5, 5+df, df)  
    B = 0.5  # window half-width in seconds
    verbose = True
    '''
    # input signal
    x = np.zeros_like(t)
    x[t < 10] = np.cos(2 * np.pi * t[t < 10])          # 1 Hz
    x[(t >= 10) & (t < 20)] = np.cos(6 * np.pi * t[(t >= 10) & (t < 20)])  # 3 Hz
    x[t >= 20] = np.cos(4 * np.pi * t[t >= 20])        # 2 Hz
    '''
    # input: linear chirp, frequency increasing from 1 Hz to 4 Hz across 30 s
    f0 = 1.0     # start frequency (Hz)
    f1 = 4.0     # end frequency (Hz)
    Tmax = 30.0  # total duration
    k = (f1 - f0) / Tmax  # frequency slope (Hz/s)

    x = np.cos(2 * np.pi * (f0 * t + 0.5 * k * t**2))

    # ===== show input signal =====
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, color='steelblue')
    plt.title('Input Signal x(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("input_signal.png", dpi=200)
    #plt.show()
    # ============================

    # calculate STFT

    # ==== Execution time start ====
    start_time = time.time()
    Y = recSTFT(dt, df, x, t, f, B, verbose)
    # ==== Execution time end =====
    comp_time = time.time() - start_time
    
    print(f"Computation time = {comp_time:.6f} s")

    # plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(Y.T), shading='auto')
    plt.title(f'Rectangular-window STFT |Y(t,f)|, dt={dt}s, df={df}s, B={B}s')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='|Y|')
    plt.tight_layout()
    plt.savefig(f"rectSTFT_fft_b{B}_dt{dt}_df{df}.png", dpi=200)
    #plt.show()
