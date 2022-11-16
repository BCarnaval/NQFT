"""Verifying 'qcm' module.
"""

from scipy.constants import pi
from pyqcm import (
    new_cluster_model,
    add_cluster,
    lattice_model,
    interaction_operator,
    hopping_operator,
    sectors,
    cluster_averages,
    new_model_instance,
    set_parameters
)
from pyqcm.spectral import mdc

from qcm import build_matrix


shape = (3, 4)
fermions_nb = 12

# Building cluster
new_cluster_model(
    name="clus",
    n_sites=shape[0] * shape[1],
    n_bath=0
)
add_cluster(name="clus", pos=[0, 0, 0], sites=build_matrix(shape))

# Initialiazing lattice using built cluster
super_vecs = [[shape[1], 0, 0], [1, shape[0], 0]]
lattice_model(name="test_qcm", superlattice=super_vecs)

# Interaction operator U
interaction_operator(name="U")

# Hopping operators (t, tp, tpp)
hopping_operator(name="t", link=[1, 0, 0], amplitude=-1)
hopping_operator(name="t", link=[0, 1, 0], amplitude=-1)

hopping_operator(name="tp", link=[1, 1, 0], amplitude=-1)
hopping_operator(name="tp", link=[-1, 1, 0], amplitude=-1)

hopping_operator(name="tpp", link=[2, 0, 0], amplitude=-1)
hopping_operator(name="tpp", link=[0, 2, 0], amplitude=-1)

# Setting target sectors
sectors(R=0, N=8, S=0)

# Setting operators parameters
set_parameters(
    """
    U = 0.5
    t = 1.0
    tp = -0.3
    tpp = 0.2
    mu = 0.000000001
    """
)

model = new_model_instance(record=True)
cluster_averages()

spectral = 1 / pi * mdc(
    freq=0.0,
    nk=400,
    eta=0.05,
    sym='RXY',
    # data_file=f'{self.model_path}/{self.spectrum_file}',
    data_file=None,
    show=True
)


if __name__ == "__main__":
    pass
