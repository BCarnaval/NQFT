"""Module permettant d'obtenir le hamiltonien du modele
de Hubbard avec un reseau a deux sites.

Tous les fermions ici traites sont exprimables dans la base de Fock
selon la sequence suivante:

 -------------------------- C_1^up
|
|   ----------------------- C_2^up
|  |
|  |   -------------------- C_1^down
|  |  |
|  |  |   ----------------- C_2^down
|  |  |  |

(--------)ket{0},

où les C_i représentent les opérateurs création et 
ket{0} l'état du vide.
"""

# import qutip
import numpy as np
from functools import reduce


vaccum = np.array([1, 0])
creation = np.array([
    [0, 0], 
    [1, 0]
    ])
destroying = creation.T # Get complex conjugate (dagger)

def get_ket(state: int):
    """Gives array-like representation of given state. 

    (This form is useful when it's time to manipulation the state using
    operators.)
    """
    binairy = format(state, 'b').zfill(4)
    ket = np.array([*binairy])

    vec_state = []
    for presence in ket:
        if int(presence):
            vec_state.append(np.matmul(creation, vaccum))
        else:
            vec_state.append(vaccum)

    return vec_state

def get_vector(ket: np.array):
    """Gives array-like 16th dimensionnal vector representation 
    of given Fock state.
    """
    return np.array(reduce(np.kron, ket))

def ops(types: list, state: int, axes: list):
    """Performs operations between (creation, destroy & number) 
    on given Fock state along specified axis.
    """
    updated = get_ket(state)
    killed_state = np.zeros((4, 2))
    fermion = np.array([0, 1])

    for type, axis in zip(types, axes):
        is_vaccum = np.dot(updated[axis], vaccum)

        if type == "create" and is_vaccum:
            updated[axis] = fermion

        elif type == "destroy" and not is_vaccum:
            updated[axis] = vaccum

        elif type == "num" and not is_vaccum:
            pass

        else:
            updated = killed_state

    return get_vector(updated)

def get_energy(model: str, states: list, U: int, **kwargs):
    """Outputs the value of energy on given eigenvector
    using the hamiltonian.
    """
    bra, ket = get_vector(get_ket(states[0])), states[1]

    if model == "Hubbard":
        t, = kwargs.values()
        H1 = ops(["num", "num"], ket, [2, 0]) + ops(["num", "num"], ket, [3, 1])

        H2 = ops(["destroy", "create"], ket, [0, 1]) + ops(["destroy",
            "create"], ket, [1, 0]) + ops(["destroy", "create"], ket, [3,
                2]) + ops(["destroy", "create"], ket, [2, 3])
        
        H = U*H1 - t*H2

    elif model == "AIM":
        mu, theta, e = kwargs.values()
        H1 = ops(["num", "num"], ket, [3, 1])

        H2 = ops(["num"], ket, [1]) + ops(["num"], ket, [3])

        H3 = ops(["destroy", "create"], ket, [1, 0]) + ops(["destroy",
            "create"], ket, [0, 1]) + ops(["destroy", "create"], ket, [3, 2]) + ops(["destroy", "create"], ket, [2, 3])

        H4 = ops(["num"], ket, [0]) + ops(["num"], ket, [2])

        H = U*H1 - mu*H2 - theta*H3 + (e - mu)*H4

    return np.dot(bra, H)

def build_H(model: str, U: int, **kwargs):
    """Build Hubbard's model hamiltonian.
    """
    H = np.zeros((16, 16))
    if model == "Hubbard":
        t, = kwargs.values()
        for i in range(16):
            for j in range(16):
                H[i, j] = get_energy(model=model, states=[i, j],
                        U=U, t=t)

    elif model == "AIM":
        mu, theta, e = kwargs.values()
        for i in range(16):
            for j in range(16):
                H[i, j] = get_energy(model=model, states=[i, j],
                        U=U, mu=mu, theta=theta, e=e)
    return H


if __name__ == "__main__":
    Hamiltonian_H = build_H(model="Hubbard", U=1, t=1)
    print(Hamiltonian_H)

    # Hamiltonian_AIM = build_H(model="AIM", U=1, mu=1, theta=1, e=1)

