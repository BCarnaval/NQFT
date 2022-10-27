"""This module is dedicated to 'pyqcm'
experimentation.
"""

import numpy as np
from pyqcm import (
    new_cluster_model,
    add_cluster,
    lattice_model,
    interaction_operator,
    hopping_operator,
    new_model_instance,
    read_cluster_model_instance,
    sectors,
    set_parameters
)


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
    >>> build_matrix(shape=(2, 2))
    >>> [[-1  0  0]
    [ 0  0  0]
    [-1  1  0]
    [ 0  1  0]]
    """
    array, idty = [], np.identity(3)
    for i in range(shape[0]):
        for j in range(shape[1]):
            elem = i * idty[1] + j * idty[0]
            array.append(elem)

    return np.array(array, dtype=np.int64)


def main(shape: tuple) -> None:
    """Main function build lattice model using 'pyqcm'.
    """
    matrix = build_matrix(shape)
    elem_nb = shape[0] * shape[1]
    bath_nb = 0

    # Building cluster
    new_cluster_model(
        "clus",
        elem_nb,
        bath_nb,
        [[i for i in range(1, elem_nb + 1 + bath_nb)]]
    )
    add_cluster("clus", [0, 0, 0], matrix)

    # Initialiazing lattice using built cluster
    lattice_model(f"2D_{elem_nb}_sites", [[4, 0, 0], [1, 3, 0]])

    # Defining operators (U, t's)
    interaction_operator("U")
    hopping_operator("t", [1, 0, 0], -1)
    hopping_operator("t", [0, 1, 0], -1)

    hopping_operator("tp", [1, 1, 0], -1)
    hopping_operator("tp", [-1, 1, 0], -1)

    hopping_operator("tpp", [2, 0, 0], -1)
    hopping_operator("tpp", [0, 2, 0], -1)

    sectors(R=0, N=10, S=0)

    # Set operators parameters
    set_parameters(
        """
            U = 8.0
            t = 1
            tp = 0.0
            tpp = 0.0
            """
    )

    # Instancing lattice model
    test_model = new_model_instance(record=True)
    test_model.print('./nqft/Data/test_model.py')
    read_cluster_model_instance(test_model)

    return


if __name__ == "__main__":
    main(shape=(3, 4))
