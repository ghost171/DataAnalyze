import numpy as np

cimport cython
cimport numpy as np

from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def euclidean_distances_cy(np.ndarray[np.float_t, ndim=2] a,
                           np.ndarray[np.float_t, ndim=2] b):
    cdef int num_a = a.shape[0]
    cdef int num_b = b.shape[0]
    cdef int dim = a.shape[1]

    cdef double d
    cdef double[:, ::1] distances = np.empty((num_a, num_b), dtype=np.float64)
 
    for i in range(num_a):
        for j in range(num_b):
            d = 0
            for k in range(dim):
                d += (a[i, k] - b[i, k]) ** 2
            distances[i, j] = sqrt(d)
            
    return np.asarray(distances)