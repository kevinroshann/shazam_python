import numpy as np

def fft_wrapper(input_data: np.ndarray) -> np.ndarray:
    """
    Wraps the recursive FFT implementation.
    Matches the logic in fft.go.
    """
    complex_array = input_data.astype(complex)
    return recursive_fft(complex_array)

def recursive_fft(x: np.ndarray) -> np.ndarray:
    """
    Computes the Fast Fourier Transform (FFT) recursively.
    Mirroring the implementation in fft.go.
    """
    n = len(x)
    if n <= 1:
        return x

    even = recursive_fft(x[0::2])
    odd = recursive_fft(x[1::2])
    
    # Precompute the twiddle factors
    t = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    
    combined = np.zeros(n, dtype=complex)
    combined[:n // 2] = even + t * odd
    combined[n // 2:] = even - t * odd
    
    return combined