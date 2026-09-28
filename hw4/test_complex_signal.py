import numpy as np
import matplotlib.pyplot as plt

def chirp(t, f0, f1):
    """線性頻率掃掠 chirp，從 f0 -> f1"""
    T = t[-1] - t[0]
    k = (f1 - f0) / T
    phase = 2*np.pi*(f0 * t + 0.5 * k * t**2)
    return np.cos(phase)

def fm_modulated(t, fc, fm, beta=1.0):
    """簡單 FM: carrier freq fc, modulating freq fm, modulation index beta"""
    return np.cos(2*np.pi*fc*t + beta * np.sin(2*np.pi*fm*t))

def am_modulated(t, carrier_freq, amp_env):
    """AM: amp_env(t) * cos(2π carrier_freq t)"""
    return amp_env * np.cos(2*np.pi*carrier_freq*t)

def generate_complex_signal(kind, t, seed=None):
    """
    kind: one of
      'multi_close'   - 多成分、頻率接近 (mode mixing)
      'chirp+noise'   - 變頻 chirp + 雜訊（非平穩）
      'am_fm'         - AM + FM 混合
      'intermittent'  - 間歇性 burst + 基底波
      'trend_nonlin'  - 非線性趨勢 + 多頻
      'hetero_noise'  - 變異數隨時間改變的雜訊 +成分
      'impulses'      - 在時間點有脈衝干擾
      'complex_all'   - 把上面幾種疊一起（最難）
    """
    rng = np.random.default_rng(seed)
    T = t[-1] - t[0]

    if kind == 'multi_close':
        # 兩個頻率非常接近 -> mode mixing 問題
        s1 = np.cos(2*np.pi*5.0*t)                  # 5 Hz
        s2 = 0.8 * np.cos(2*np.pi*5.4*t + 0.5)      # 5.4 Hz
        s3 = 0.5 * np.cos(2*np.pi*12.0*t)           # 12 Hz
        noise = 0.2 * rng.normal(size=len(t))
        return s1 + s2 + s3 + noise

    if kind == 'chirp+noise':
        # 長時段線性 chirp (低頻 -> 高频) + colored noise
        s = chirp(t, f0=1.0, f1=40.0)
        # colored noise: 1/f-like by filtering white noise in frequency domain (simple)
        w = rng.normal(size=len(t))
        W = np.fft.rfft(w)
        freqs = np.fft.rfftfreq(len(t), d=t[1]-t[0])
        # avoid divide by zero
        W = W / (1 + freqs**0.7)
        colored = np.fft.irfft(W, n=len(t))
        colored = colored / np.std(colored) * 0.5
        return s + colored

    if kind == 'am_fm':
        # AM envelope slowly varying + FM carrier
        env = 1.0 + 0.6 * np.sin(2*np.pi*0.2*t) + 0.2*np.sin(2*np.pi*0.05*t)
        carrier_f = 10.0
        fm = fm_modulated(t, fc=carrier_f, fm=0.5, beta=4.0)
        am = am_modulated(t, carrier_f, amp_env=env)
        # combine AM and FM components + a lowfreq component
        return 0.6*fm + am + 0.3*np.cos(2*np.pi*2.0*t)

    if kind == 'intermittent':
        # 間歇性 burst: 在幾段時間內出現高頻成分
        base = 0.3*np.cos(2*np.pi*3*t) + 0.2*np.cos(2*np.pi*7*t)
        bursts = np.zeros_like(t)
        # define bursts centers and widths
        centers = [T*0.2, T*0.5, T*0.75]
        for c in centers:
            width = 0.5  # seconds
            mask = np.exp(-((t - c)**2) / (2*(width**2)))
            bursts += mask * (1.5 * np.cos(2*np.pi*40*t))  # 40 Hz bursts
        noise = 0.1 * rng.normal(size=len(t))
        return base + bursts + noise

    if kind == 'trend_nonlin':
        # 非線性趨勢 (多項式 + 突變) + 幾個成分
        trend = 0.02*(t**2) - 0.5*np.sin(0.1*t)
        comps = 0.8*np.cos(2*np.pi*2*t + 0.2*t) + 0.5*np.cos(2*np.pi*8*t)
        # 突變 jump
        jump_time = T*0.6
        jump = np.where(t > jump_time, 1.0, 0.0)
        noise = 0.2 * rng.normal(size=len(t))
        return trend + comps + 0.5*jump + noise

    if kind == 'hetero_noise':
        # noise variance increases with time (heteroscedastic)
        base = np.cos(2*np.pi*4*t) + 0.6*np.cos(2*np.pi*9*t)
        sigma = 0.05 + 0.5 * (t / T)  # grows from 0.05 -> 0.55
        noise = rng.normal(scale=sigma)
        return base + noise

    if kind == 'impulses':
        base = 0.5*np.cos(2*np.pi*3*t) + 0.3*np.cos(2*np.pi*7*t)
        y = base.copy()
        # random impulses
        n_imp = 8
        locs = rng.choice(len(t), size=n_imp, replace=False)
        for idx in locs:
            amp = rng.normal(loc=2.0, scale=0.5)
            width = int(0.005 * len(t))  # short width relative to total length
            lo = max(0, idx - width//2)
            hi = min(len(t), idx + width//2)
            y[lo:hi] += amp * np.hanning(hi-lo)
        y += 0.1 * rng.normal(size=len(t))
        return y

    if kind == 'complex_all':
        # 最複雜：trend + multi close + chirp + intermittent bursts + hetero noise
        s = generate_complex_signal('trend_nonlin', t, seed=seed)
        s += 0.8 * generate_complex_signal('multi_close', t, seed=seed+1 if seed is not None else None)
        s += 0.6 * chirp(t, 0.5, 30)
        s += 0.5 * generate_complex_signal('intermittent', t, seed=seed+2 if seed is not None else None)
        # add hetero noise
        s += 0.3 * generate_complex_signal('hetero_noise', t, seed=seed+3 if seed is not None else None)
        return s

    raise ValueError("Unknown kind")

if __name__ == "__main__":
    # 範例：建一個時間軸
    fs = 100.0   # 100 Hz sampling
    t = np.arange(0, 10, 1/fs)

    # 你可以用這列測試各種訊號
    kinds = ['multi_close', 'chirp+noise', 'am_fm', 'intermittent',
            'trend_nonlin', 'hetero_noise', 'impulses', 'complex_all']

    plt.figure(figsize=(12,8))
    for i, k in enumerate(kinds[:4]):   # 顯示前 4 種作示範
        plt.subplot(4,1,i+1)
        s = generate_complex_signal(k, t, seed=42)
        plt.plot(t, s)
        plt.title(k)
        plt.xlim(0, 10)
    plt.tight_layout()
    plt.show()
