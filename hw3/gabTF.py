import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import wavfile

def gabFT(x, tau, t, f, sgm):
    n0 = 0
    T = len(tau)
    c0 = 0
    C = len(t)
    F = len(f)
    dt = t[1]-t[0]
    dtau = tau[1]-tau[0]
    df = f[1]-f[0]
    N = int(1/(dtau * df))
    S = int(dt/dtau)

    m0 = f[0]/df
    m = np.arange(m0, m0+F, 1)

    # calculate Gaussian window
    a = 1.4143 / np.sqrt(sgm)    
    Q = int(np.round(a / dtau))   # half-width in samples
    w = np.arange(-Q, Q+1) * dtau  # length = 2*Q+1, perfectly symmetric
    g = np.exp(-sgm * np.pi * w**2)

    Y = np.zeros((C, F), dtype=complex)

    for n in range(C):
        # n will be 0, 1, ... C-1, assume c0 = 0
        seg = np.zeros(2 * Q + 1, dtype=complex)
        start = n*S - Q
        end = n*S + Q + 1
        valid_start = max(start, 0)
        valid_end = min(end, len(x))
        seg_start = valid_start - start
        seg_end = seg_start + (valid_end - valid_start)
        seg[seg_start:seg_end] = x[valid_start:valid_end]
        x1 = np.zeros(N, dtype=complex)
        x1[:len(seg)] = seg*g
        X1 = np.fft.fft(x1, N)
        # phase correction
        phase_corr = np.exp(1j * 2 * np.pi * (Q - n*S) * m / N)
        m1 = m % N
        m1 = m1.astype(int)
        Y[n, :] = dtau * phase_corr * X1[m1]

    Y = Y * (sgm ** 0.25)

    return Y
#=====main=====
if __name__ == "__main__":
    # read wav file
    fs, a1 = wavfile.read('Chord.wav')
    dtau = 1/fs
    # take the first channel and turn to float
    x = a1[:, 0].astype(float)

    # print input
    print("fs:", fs)
    print("dtau:", dtau)
    print("x:",x)
    
    tau = np.arange(0, len(x)) / fs
    n0 = 0
    T = len(tau)
    print("n0:", n0)
    print("T:", T)
    print("tau:", tau)
    print("end of time:", tau[-1])

    # ===== show input signal =====
    plt.figure(figsize=(10, 4))
    plt.plot(tau, x, color='steelblue')
    plt.title('Input Signal x(tau)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("input_signal.png", dpi=200)
    #plt.show()
    # ============================

    dt = 0.01
    S = dt/dtau
    t = np.arange(0, tau[-1], dt)
    c0 = 0
    C = len(t)
    print("S:", S)
    print("C:", C)
    #print("t:", t)

    df = 1
    f = np.arange(20, 1000+df, df)
    m0 = 0
    F = len(f)
    print("F:", F)
    #print("f:", f)
    N = int(1/(dtau * df))
    print("N:", N)

    sgm = 200
    a = 1.4143 / np.sqrt(sgm)    
    Q = int(np.round(a / dtau))   # half-width in samples
    w = np.arange(-Q, Q+1) * dtau  # length = 2*Q+1, perfectly symmetric
    g = np.exp(-sgm * np.pi * w**2)
    print("Q:", Q)
    print("g:", g)
    print("len of g:", len(g))

    # ==== Execution time start ====
    start_time = time.time()
    Y = gabFT(x = x, tau = tau, t = t, f = f, sgm = sgm)
    # ==== Execution time end =====
    comp_time = time.time() - start_time

    print(f"Computation time = {comp_time:.6f} s")

    # plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(Y.T), shading='auto')
    plt.title(f'Gabor Transform of |Y(t,f)|, dt={dt}s, df={df}s')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='|Y|')
    plt.tight_layout()
    plt.savefig(f"GaborTransform.png", dpi=200)
    #plt.show()