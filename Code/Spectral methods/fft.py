import numpy as np

def dft(x):
    """
    Compute the discrete Fourier Transform of the 1D array x
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def fft(x):
    """
    Compute the Fast Fourier Transform of the 1D array x
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]

    if N <= 1:
        return x

    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + T[:N // 2] * odd,
                           even + T[N // 2:] * odd])


def ifft(X):
    """
    Compute the Inverse Fast Fourier Transform of the 1D array X
    """
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]

    if N <= 1:
        return X

    x = fft(np.conj(X))
    return np.conj(x) / N

    