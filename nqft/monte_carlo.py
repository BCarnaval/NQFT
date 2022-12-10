import numpy as np
from scipy.signal import convolve2d


def build_h(shape: tuple[int], mu: float, hops: np.array) -> np.ndarray:
    """Docs
    """
    t, tp, tpp = hops
    sites = shape[0] * shape[1]
    h = np.ones(shape=(sites, sites))
    t_ij = np.array([
        [0, 0, tpp, 0, 0],
        [0, tp, t, tp, 0],
        [tpp, t, mu, t, tpp],
        [0, tp, t, tp, 0],
        [0, 0, tpp, 0, 0]
    ])

    h = convolve2d(h, t_ij, mode='same', boundary='wrap')
    # for coord_1, i in np.ndenumerate(cluster):
    #     x1, y1 = coord_1
    #     for coord_2, j in np.ndenumerate(cluster):
    #         x2, y2 = coord_2
    #         dist = ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)
    #         if dist > 2.0:
    #             h[i, j] = 0.0
    #         elif dist == 2.0:
    #             h[i, j] = hops[2]
    #         elif dist == (2.0)**(1/2):
    #             h[i, j] = hops[1]
    #         elif dist == 1.0:
    #             h[i, j] = hops[0]

    return h


if __name__ == "__main__":
    # Variables & Parameters
    shape = (3, 3)
    sites = shape[0] * shape[1]
    hoppings = np.array([1, 2, 3])

    # Building Hamiltonian
    H = build_h(shape=shape, mu=0.0, hops=hoppings)
    print(H)
