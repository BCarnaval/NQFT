from pyqcm import *
from pyqcm.draw_operator import *


def build_matrix(shape: tuple) -> list:
    """Gives a coordinates matrix of a cluster having
    shape[0]*shape[1] sites.

    Parameters
    ----------
    shape: tuple, shape (2, 1), default=None
        Shape of sites network.

    Returns
    -------
    array: np.ndarray, shape (*shape)
        Numpy array of coordinates.
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

    new_cluster_model('clus', elem_nb, 0, [[i for i in range(1, elem_nb + 1)]])
    add_cluster('clus', [0,0,0], matrix)

    lattice_model(f'2D_{elem_nb}', [[elem_nb,0,0]])

    interaction_operator('U')
    hopping_operator('t', [1,0,0], -1)
    draw_operator('U')

    return


if __name__ == "__main__":
    main(shape=(4,4))

