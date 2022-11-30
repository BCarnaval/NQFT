import numpy as np


def build_h(shape: tuple[int], eps: np.array, hops: np.array) -> np.ndarray:
    """Docs
    """
    epsilons = np.diag(eps)
    sites, it = shape[0] * shape[1], 0
    h = np.zeros(shape=(sites, sites))
    i, j = np.indices(dimensions=shape, sparse=True)
    for idx_x in range(shape[0]):
        for idx_y in range(shape[1]):
            h[it] = np.sqrt((i - idx_x)**2 + (j - idx_y)**2).ravel()
            it += 1

    return h + epsilons


if __name__ == "__main__":
    shape = (2, 2)
    hoppings = np.array([1, 2, 3])
    epsilons = np.array([5, 5, 5, 5])
    H = build_h(shape=shape, eps=epsilons, hops=hoppings)

    print(H)


"""
Example 2x2 cluster:

[0 1
 2 3]

[e1 t t tp
 t e2 tp t
 t tp e3 t
 tp t t e4]

Example 3x3 cluster:

[0 1 2
 3 4 5
 6 7 8]

[e 1 3 1 2 2 3 0 0
 1 e 1 2 1 2 0 3 0
 3 1 e 2 2 1 0 0 3
 1 2 2 e 1 3 1 2 1
 2 1 2 1 e 1 2 1 2
 2 2 1 3 1 e 2 1 1
 3 0 0 1 2 2 e 1 3
 0 3 0 2 1 2 1 e 1
 0 0 3 2 2 1 3 1 e]
"""
