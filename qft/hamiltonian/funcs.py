# /usr/bin/env python3

"""Module de test des fonctionnalités de la librairie
Python Qutip. Les états ici discutés possèdent la forme

  --------------------------------------- (c^dagger_1)^n_1 up
 |
 |   ------------------------------------ (c^dagger_2)^n_2 up
 |  |                                             .
 |  |                                             .
 |  |                                             .
 |  |                                       
 |  |         --------------------------- (c^dagger_1)^n_1 down
 |  |        |
 |  |        |   ------------------------ (c^dagger_2)^n_2 down
 |  |        |  |                                 .
 |  |        |  |                                 .
 |  |        |  |                                 .
(      ...        ...)ket{0}
"""

import numpy as np
from tqdm import trange
from numpy.random import randint
from qutip import (Qobj, basis, create, destroy, num, tensor, identity)


def scalar(m: Qobj, n=None) -> float:
    """Computes scalar product for Fock space vectors such as

    scalar(m, n) = < m | n >.

    Parameters
    ----------
    m: qutip.Qobj, default=None
        Bra on which perform scalar product.
    n: qutip.Qobj, default=None
        Ket on which perform scalar product.

    Returns
    -------
    -: int, float
        Result of scalar product.
    """
    if n:
        val = m.dag()*n
    else:
        val = m.dag()*m
    return val.tr()

def delta(j: int, k: int) -> float:
    """Kronecker delta function.

    Parameters
    ----------
    j: int, default=None
        First indice.
    k: int, default=None
        Second indice.

    Returns
    -------
    -: float (0.0 or 1.0)
    """
    return 1.0 if j == k else 0.0


class Network():
    """Docs
    """

    def __init__(self, sites_nb: int) -> None:
        """Docs
        """
        self.sites = sites_nb
        self.vaccum = basis(2)
        self.creation = create(2)
        self.anihilation = destroy(2)
        self.number = num(2)
        self.I = identity(2)
        return

    def get_state(self, state: int, type="ket") -> Qobj:
        """Gives array-like representation of given state using 
        Qutip 'tensor' function.

        Parameters
        ----------
        state: int, default=None
            Integer representing state number in Fock space.
        type: str, default='ket'
            Type of vector outputed (bra, ket)

        Returns
        -------
        bra, ket: qutip.Qobj, shape (4^SITES, 1)
            Full vector representation.
        """
        binairy = format(state, 'b').zfill(self.sites*2)
        ket = [*binairy]

        vec_state = []
        for presence in ket:
            if int(presence):
                vec_state.append(self.creation*self.vaccum)
            else:
                vec_state.append(self.vaccum)

        vec = tensor(*vec_state)

        return vec.dag() if type == "bra" else vec


    def get_hamiltonian(self, model: str, U: int, **kwargs) -> Qobj:
        """Outputs the hamiltonian of specified many-body model.

        Parameters
        ----------
        model: str, default=None
            Fermions network configuration.
        U: int, default=None
            Module of interaction between fermions.
        **kwargs:
            t: int, default=None
                Probability amplitude for fermions to jump.

        Returns
        -------
        H: qutip.Qobj, shape (4^SITES, 4^SITES)
            Tensor object representing hamitonian for given 'model'.
        """
        if model == "Hubbard":
            H1, H2 = 0, 0
            t, = kwargs.values()

            for idx in trange(self.sites, desc="Hamitonian"):
                sites = [i for i in range(self.sites)]
                ops_1 = [self.I for i in range(self.sites*2)]

                ops_1[idx] = self.number
                ops_1[self.sites + idx] = self.number

                sites.remove(idx)
                H1 += tensor(*ops_1)

                for ext in sites:
                    ops_2_up = [self.I for i in range(self.sites*2)]
                    ops_2_down = [self.I for i in range(self.sites*2)]

                    ops_2_up[idx] = self.creation
                    ops_2_up[ext] = self.anihilation

                    ops_2_down[self.sites + idx] = self.creation
                    ops_2_down[self.sites + ext] = self.anihilation

                    H2 += tensor(*ops_2_up) + tensor(*ops_2_down)

            H = U*H1 - t*H2

        elif model != "Hubbard":
            pass

        return H

    def get_e(self, H: Qobj, states: list) -> float:
        """Outputs a matrix element (energy) from the hamiltonian using 
        given kets on which perform projection.

        Parameters
        ----------
        model: str, default=None
            Fermions network configuration.
        states: array-like, shape (2, 1), default=None
            Vectors used to process scalar product on H.
            (ex: states=[bra, ket] as integers to convert from binairy)

        Returns
        -------
        E: int
            Representation of projected vectors on H (energy).
        """
        bra, ket = self.get_state(states[0], type="bra"), self.get_state(states[1])
        E = bra*H*ket

        return E.tr()

    def lanczos(self, H: Qobj, iterations: int, init_state=None) -> tuple:
        """Docs
        """
        if not init_state:
            dim = H.shape[0]
            state = randint(dim - 1)
            init = self.get_state(state=state)
        else:
            init = init_state

        H_n = np.zeros((iterations, iterations), dtype=np.complex64)
        for iter in trange(iterations, desc="Lanczos algorithm"):
            if iter == 0:
                a_n = H.matrix_element(init.dag(), init) / scalar(init)
                b_n = 0

                phi_n_plus = H*init - a_n*init
                phi_n_minus = init
                phi_n = phi_n_plus
            else:
                a_n_minus = a_n
                a_n = H.matrix_element(phi_n.dag(), phi_n) / scalar(phi_n)

                b_n_minus = b_n
                b_n = np.sqrt(scalar(phi_n) / scalar(phi_n_minus))

                phi_n_plus = H*phi_n - a_n*phi_n - b_n**2*phi_n_minus
                phi_n_minus = phi_n
                phi_n = phi_n_plus
                
                for lgn in range(iter + 1):
                    H_n[lgn, iter - 1] = b_n*delta(lgn, iter) + a_n_minus*delta(lgn, iter - 1) + b_n_minus*delta(lgn, iter - 2)

                    H_n[lgn, iter] = a_n*delta(lgn, iter) + b_n*delta(lgn, iter - 1)

            H_phi = Qobj(H_n)

            if scalar(phi_n) == 0:        
                return H_phi.eigenenergies()[-1], H_phi.eigenstates()[-1][-1]
            else:
                pass

        print(f"Lanczos algorithm hasn't converged with {iterations} iterations!\n")
        return 0.0, basis(1)

if __name__ == "__main__":
    N = Network(sites_nb=8)
    H = N.get_hamiltonian(model="Hubbard", U=1, t=1)
    val, state = N.lanczos(H=H, iterations=10)

