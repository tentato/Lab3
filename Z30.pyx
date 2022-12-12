from cython.parallel import prange
import numpy as np
cimport numpy as np

np.import_array()

DTYPE = int

ctypedef np.int_t DTYPE_t

def seq_convolve(np.ndarray image, np.ndarray kernel):

    assert image.dtype == DTYPE and kernel.dtype == DTYPE

    cdef int img_h = image.shape[0]
    cdef int img_w = image.shape[1]
    cdef int kernel_h = kernel.shape[0]
    cdef int kernel_w = kernel.shape[1]
    cdef int kernel_mid_h_idx = kernel_h // 2
    cdef int kernel_mid_w_idx = kernel_w // 2
    cdef int i_max = img_h + 2 * kernel_mid_h_idx
    cdef int j_max = img_w + 2 * kernel_mid_w_idx
    
    cdef np.ndarray result = np.zeros([i_max, j_max], dtype=DTYPE)
    cdef int x, y, s, t, v, w
    cdef int hernel_h1, kernel_h2, kernel_w1, kernel_w2
    cdef DTYPE_t res_value

    for x in range(i_max):
        for y in range(j_max):
            hernel_h1 = max(kernel_mid_h_idx - x, -kernel_mid_h_idx)
            kernel_h2 = min((i_max - x) - kernel_mid_h_idx, kernel_mid_h_idx + 1)
            kernel_w1 = max(kernel_mid_w_idx - y, -kernel_mid_w_idx)
            kernel_w2 = min((j_max - y) - kernel_mid_w_idx, kernel_mid_w_idx + 1)
            res_value = 0
            for s in range(hernel_h1, kernel_h2):
                for t in range(kernel_w1, kernel_w2):
                    v = x - kernel_mid_h_idx + s
                    w = y - kernel_mid_w_idx + t
                    res_value += kernel[kernel_mid_h_idx - s, kernel_mid_w_idx - t] * image[v, w]
            result[x, y] = res_value
    return result