import numpy as np

from memory_profiler import profile


@profile
def euclidean_distances(a, b):
    a_norm = (a ** 2).sum(axis=1, keepdims=True) 
    b_norm = (b ** 2).sum(axis=1)[np.newaxis, :]
    return np.sqrt(
        -2 * np.dot(a, b.T)
        + a_norm
        + b_norm
    )


if __name__ == '__main__':
    rnd = np.random.RandomState(9876)

    a = rnd.random(size=(2000, 100))
    b = rnd.random(size=(1000, 100))
    c = euclidean_distances(a, b)
