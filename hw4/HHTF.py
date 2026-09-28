import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.interpolate as interpolate

try:
    from test_complex_signal import generate_complex_signal
    print("Successfully import complex signal")
    complexFlag = True
except ImportError:
    complexFlag = False
    print("Cannot find complex signal module")

np.set_printoptions(threshold=10)

def show_signal(x, t, title="Input Signal", y_max = 3, save_fig = False):
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, color='steelblue')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.ylim(-y_max, y_max)
    plt.grid(True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{title}.png", dpi=200)
    plt.show()
    plt.close()
    return

def find_extreme(x, t, endpoint = True):
    dx = np.diff(x)
    print(len(dx))
    print(dx)
    signs = np.sign(dx)
    sign_change = np.diff(signs)
    max_idx = np.where(sign_change < 0)[0] + 1
    min_idx = np.where(sign_change > 0)[0] + 1

    # For end points:
    extreme_idx = np.sort(np.concatenate([max_idx, min_idx]))
    if(endpoint):
        endpoints = [0, len(x) - 1]

        # For right endpoint: x[0]
        d10 = abs(extreme_idx[0])
        d21 = abs(extreme_idx[1]-extreme_idx[0])

        if d10 > 0.5 * d21:
            # as extreme
            if x[0] > x[1]:
                max_idx = np.insert(max_idx, 0, 0)
            else:
                min_idx = np.insert(min_idx, 0, 0)
        # For left endpoint: x[len(x)-1]
        d10_ = abs(len(x)-1-extreme_idx[-1])
        d21_ = abs(extreme_idx[-1]-extreme_idx[-2])

        if d10_ > 0.5 * d21_:
            # as extreme
            if x[-1] > x[-2]:
                max_idx = np.append(max_idx, len(x)-1)
            else:
                min_idx = np.append(min_idx, len(x)-1)

    return max_idx, min_idx, extreme_idx

def find_upper_bottom_envelop(x, t):
    dt = t[1]-t[0]
    local_max, local_min, _ = find_extreme(x, t)
    print("local max:", local_max)
    print("local_min:", local_min)
    
    if len(local_max) > 3:
        t_, c, k = interpolate.splrep(local_max * dt, x[local_max], k = 3) # k: degree
        f = interpolate.BSpline(t_, c, k)
        emax = f(t)
    else:
        emax = np.interp(t, t[local_max], x[local_max])
    #show_signal(emax, t, "Local peaks")
    if len(local_min) > 3:
        t_, c, k = interpolate.splrep(local_min * dt, x[local_min], k = 3)
        f = interpolate.BSpline(t_, c, k)
        emin = f(t)
    else:
        emin = np.interp(t, t[local_min], x[local_min])
    return emax, emin, local_max, local_min

def checkIMF(x, t, threshold):
    u1, u0, local_max, local_min = find_upper_bottom_envelop(x, t)
    posflag = np.all(x[local_max]>0)
    if not posflag:
        return False
    negflag = np.all(x[local_min]<0)
    if not negflag:
        return False
    lessflag = np.all(np.abs((u1+u0)/2)<threshold)
    if not lessflag:
        return False
    return True

def hht(x, t, thr, K = 10):
    c = [] # list to store IMP
    findtrend = False
    n = 0
    y = x
    while not findtrend:
        print(f"n={n}")
        isIMF = False
        k = 0
        while (not isIMF):
            print(f"k={k}")
            emax, emin, _, _ = find_upper_bottom_envelop(y, t)
            z = (emin+emax)/2
            h = y - z
            if(checkIMF(h, t, thr)):
                isIMF = True
            else:
                y = h
                k = k + 1
        c.append(h) # Append (n+1)-th IMF
        x0 = x - sum(c)
        _, _, extreme_index = find_extreme(x0, t, endpoint=False) #Not include endpoint
        if len(extreme_index) <= 3: # trend should be polynomial of order <= 4
            findtrend = True
        else:
            n = n + 1
            y = x0
    return c, x0

# ===== main =====
if __name__ == "__main__":
    dt = 0.01 # Δt = 0.05
    t = np.arange(0, 10+dt, dt)  
    verbose = True
    thr = 0.2
    x = 0.2 * t + np.cos(2 * np.pi * t) + 0.4 * np.cos(10 * np.pi * t)
    #kinds = ['multi_close', 'chirp+noise', 'am_fm', 'intermittent',
    #     'trend_nonlin', 'hetero_noise', 'impulses', 'complex_all']
    #x = generate_complex_signal(kinds[7],t,seed=42)
    # ===== show input signal =====
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, color='steelblue')
    plt.title('Input Signal x(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("input_signal.png", dpi=200)
    plt.show()
    plt.close()
    # ============================

    # calculate Hilbert Huang Transform

    # ==== Execution time start ====
    start_time = time.time()
    IMFs, x0 = hht(x, t, thr)
    # ==== Execution time end =====
    comp_time = time.time() - start_time
    n = 1
    for imf in IMFs:
        show_signal(imf, t, f"IMF{n}", save_fig = True)
        n = n + 1
    show_signal(x0, t, f"Trend-x0(t)", save_fig = True)

    print(f"Computation time = {comp_time:.6f} s")
