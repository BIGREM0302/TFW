import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# ----------------------
# Daubechies 10-point case
# ----------------------
# Orthonormal filter:
g = [0.0033, -0.0126, -0.0062, 0.0776, -0.0322, -0.2423, 0.1384, 0.7243, 0.6038, 0.1601] #n=0, 1, ..., 9
# hn = (-1)^n*g[-n-k], k=-9
h = [0.1601, -0.6038, 0.7243, -0.1384, -0.2423, 0.0322, 0.0776, 0.0062, -0.0126, -0.0033] #n=0, 1, ..., 9
# g1n = g[-n]
g1 = [0.1601, 0.6038, 0.7243, 0.1384, -0.2423, -0.0322, 0.0776, -0.0062, -0.0126, 0.0033] #n=-9, -8, ..., 0
# h1n = h[-n]
h1 = [-0.0033, -0.0126, 0.0062, 0.0776, 0.0322, -0.2423, -0.1384, 0.7243, -0.6038, 0.1601] #n=-9, -8, ..., 0

goffset = 0
hoffset = 0
g1offset = -9
h1offset = -9

def im(x):
    return Image.fromarray(np.uint8(x))

def downsample(x, axis, Q, start=0):
    slicer = [slice(None)] * x.ndim
    slicer[axis] = slice(start, None, Q)
    return x[tuple(slicer)]

def upsample(x, axis, Q):
    outshape = list(x.shape)
    outshape[axis] *= Q
    y = np.zeros(outshape, dtype=x.dtype)
    slicer = [slice(None)] * x.ndim
    slicer[axis] = slice(0, None, Q)
    y[tuple(slicer)] = x
    #print('upsample result shape:', y.shape)
    return y

def convolution_block(x, k, axis, k_start):
    """
    Exactly equivalent to:
    y[n] = sum_j x[n - (j + k_start)] * k[j]
    Output length = N + L - 1
    """
    k = np.asarray(k, dtype=float)

    def conv1d(sig):
        return np.convolve(sig, k, mode='full')

    return np.apply_along_axis(conv1d, axis, x)

def _convolution_block(x, k, axis, k_start):
    # convolution y[n] = sum_k(x[k]k[n-k])
    # should consider filter's index
    k = np.asarray(k)
    M, N = x.shape
    L = k.shape[0]
    outlen = x.shape[axis] + L - 1
    outshape = list(x.shape)
    outshape[axis] = outlen
    y = np.zeros(outshape, dtype=float)
    # move axis to last for simpler indexing
    x_move = np.moveaxis(x, axis, -1)
    y_move = np.moveaxis(y, axis, -1)

    for idx in np.ndindex(x_move.shape[:-1]):
        row = x_move[idx] # 1D signal
        for n in range(outlen):
            acc = 0.0
            for j in range(L):
                x_idx = n - (j + k_start)
                if 0 <= x_idx < row.shape[0]:
                    acc += row[x_idx]*k[j]
            y_move[idx + (n,)] = acc

    return np.moveaxis(y_move, -1, axis)

def wavedbc10(x):
    print("Start 2D-DWT")
    v1L = downsample(convolution_block(x, g, axis=1, k_start=goffset), axis=1, Q=2)
    v1H = downsample(convolution_block(x, h, axis=1, k_start=hoffset), axis=1, Q=2)

    x1L = downsample(convolution_block(v1L, g, axis=0, k_start=goffset), axis=0, Q=2)
    x1H1 = downsample(convolution_block(v1L, h, axis=0, k_start=hoffset), axis=0, Q=2)
    x1H2 = downsample(convolution_block(v1H, g, axis=0, k_start=goffset), axis=0, Q=2)
    x1H3 = downsample(convolution_block(v1H, h, axis=0, k_start=hoffset), axis=0, Q=2)
    print("Complete 2D-DWT")
    return x1L, x1H1, x1H2, x1H3

def iwavedbc10(x1L, x1H1, x1H2, x1H3):
    print("Start inverse 2D-DWT")
    v2L = convolution_block(upsample(x1L, axis=0, Q=2), g1, axis=0, k_start=g1offset) + convolution_block(upsample(x1H1, axis=0, Q=2), h1, axis=0, k_start=h1offset)
    v2H = convolution_block(upsample(x1H2, axis=0, Q=2), g1, axis=0, k_start=g1offset) + convolution_block(upsample(x1H3, axis=0, Q=2), h1, axis=0, k_start=h1offset)
    print("v2L shape:", v2L.shape)
    x = convolution_block(upsample(v2L, axis=1, Q=2), g1, axis=1, k_start=g1offset) + convolution_block(upsample(v2H, axis=1, Q=2), h1, axis=1, k_start=h1offset)
    print("Complete inverse 2D-DWT")
    return x

def stack_results(x1L, x1H1, x1H2, x1H3):
    x1L = im(x1L)
    x1H1 = im(x1H1)
    x1H2 = im(x1H2)
    x1H3 = im(x1H3)
    x1L.save("x1L.png")
    x1H1.save("x1H1.png")
    x1H2.save("x1H2.png")
    x1H3.save("x1H3.png")
    imgs = [x1L, x1H1, x1H2, x1H3]
    w, h = imgs[0].size
    canvas = Image.new("RGB", (w * 2, h * 2))
    canvas.paste(x1L,  (0, 0))
    canvas.paste(x1H1, (0, h))
    canvas.paste(x1H2, (w, 0))
    canvas.paste(x1H3, (w, h))
    canvas.save("dwt2D_result.png")


def main(img_path):
    img = Image.open(img_path).convert('L')
    img.save("gray_scale.png")
    x = np.array(img) # x[m,n]
    H, W = x.shape
    print(x)
    print('shape:', x.shape)
    x1L, x1H1, x1H2, x1H3 = wavedbc10(x)
    print('shape of 4 results:', x1L.shape, x1H1.shape, x1H2.shape, x1H3.shape)
    stack_results(x1L, x1H1, x1H2, x1H3)
    
    x_ = iwavedbc10(x1L, x1H1, x1H2, x1H3)
    #x_ = x_[:H, :W]
    print('shape of reconstruction:', x_.shape)
    recon_img = im(x_)
    recon_img.save("reconstructed_image.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discrete Wavelet Transform")

    parser.add_argument(
        "img_path",
        nargs="?",
        default="test.jpg",
        help="Image path (default: test.jpg)"
    )
    args = parser.parse_args()
    main(args.img_path)
