import numpy as np
import numpy.linalg as npla
from numba import jit, complex128, float64

@jit(complex128[::1](complex128[:,::1], complex128[:,::1], complex128[::1], complex128[::1], complex128[:,::1], complex128[::1]), nopython=True)
def noise(L, jump_op, ss, pops, Q, freq):
    noise = np.zeros(freq.size, dtype=complex128)
    for i in range(len(freq)):
        R_plus = np.dot(Q, np.dot(npla.pinv(1.j*freq[i]*np.eye(L.shape[0])-L), Q))
        R_minus = np.dot(Q, np.dot(npla.pinv(-1.j*freq[i]*np.eye(L.shape[0])-L), Q))
        noise[i] = np.dot(pops, np.dot(jump_op, ss)) \
                        + np.dot(pops, np.dot(np.dot(np.dot(jump_op, R_plus), jump_op) \
                                                   + np.dot(np.dot(jump_op, R_minus), jump_op), ss))
    return noise

@jit(nopython=True)
def skewness():
    pass