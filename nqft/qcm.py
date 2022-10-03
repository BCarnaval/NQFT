"""This module is dedicated to 'pyqcm'
experimentation.
"""

from pyqcm import *
from pyqcm.cdmft import cdmft
from pyqcm.draw_operator import draw_operator
from pyqcm.spectral import plot_dispersion, G_dispersion


def build_matrix(shape: tuple) -> list:
    """Gives a coordinates matrix of a cluster having
    shape[0]*shape[1] sites.

    Parameters
    ----------
    shape: tuple, shape=(2, 1), default=None
        Shape of sites network.

    Returns
    -------
    array: np.ndarray, shape=(*shape)
        Numpy array of coordinates.

    Examples
    --------
    >>> build_matrix(shape=(2,2))
    >>> [[-1  0  0]
    [ 0  0  0]
    [-1  1  0]
    [ 0  1  0]]
    """
    array, I = [], np.identity(3)
    for i in range(shape[0]):
        for j in range(shape[1]):
            elem = -I[0] + j*I[0]
            array.append(elem + i*I[1])

    return np.array(array, dtype=np.int64)

def main(shape: tuple) -> None:
    """Main function calls other ones.
    """
    matrix = build_matrix(shape)
    elem_nb = shape[0]*shape[1]
    bath_nb = 0

    new_cluster_model('clus',
                      elem_nb,
                      bath_nb,
                      [[i for i in range(1, elem_nb + 1 + bath_nb)]]
                      )

    add_cluster('clus', [0, 0, 0], matrix)

    lattice_model(f'2D_{elem_nb}', [[2, 0, 0], [0, 2, 0]])

    interaction_operator('U', link=([0, 1, 0]))
    hopping_operator('t', [1, 0, 0], -1)

    sectors(R=0, N=4, S=0)

    set_parameters(
            """
            U = 1
            t = 1
            """
            )

    plot_dispersion(quadrant=False)
    return


if __name__ == "__main__":
    main(shape=(2, 2))

