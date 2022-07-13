import numpy as np

from memory_profiler import profile


@profile
def euclidean_distances(a, b):
    a = a[:, np.newaxis, :]
    b = b[np.newaxis, :, :]
    return (
        (a - b) ** 2
    ).sum(axis=-1)


if __name__ == '__main__':
    rnd = np.random.RandomState(9876)

    a = rnd.random(size=(2000, 100))
    b = rnd.random(size=(1000, 100))
    c = euclidean_distances(a, b)
