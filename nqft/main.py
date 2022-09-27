"""This module tests Qutip python library functions, objects and
attributes in the context of Hubbard model using square fermion
networks.
"""

import numpy as np
from rich.progress import track
from qutip import Qobj, basis, create, destroy, num, tensor, identity

from hamiltonian._functions import scalar, delta


class Network():
    """A class representing a fermionic many-body problem Network

    Attributes
    ----------
    sites_nb: int, default=None
        Number of sites of the network
    vaccum: Qobj, shape (2, 1)
        Vaccum state of 2D Fock space.
    creation: Qobj, shape (2, 2)
        Creation operator in second quantization formalism.
    anihilation: Qobj, shape (2, 2)
        Anihilation operator in second quantization formalism.
    number: Qobj, shape (2, 2)
        Number operator in second quantization formalism.
    I: Qobj, shape (2, 2)
        Identity operator.
    """

    def __init__(self, sites_nb: int) -> None:
        """Sets Network attributes to given values and computes
        the other ones.
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
        Qutip 'tensor' operation.

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

        Examples
        --------
        >>> N = Network(sites_nb=1)
        >>> self.get_state(state=2)
        >>> Quantum object: dims = [[2, 2], [1, 1]], shape = (4, 1), type = ket
        Qobj data =
        [[0.]
         [0.]
         [1.]
         [0.]]
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
        """Outputs the hamiltonian of fermion network using
        specified many-body model.

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

        Examples
        --------
        >>> N = Network(sites_nb=1)
        >>> N.get_hamiltonian(model="Hubbard", U=1, t=1)
        >>> Quantum object: dims = [[2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2]],
        shape = (256, 256), type = oper, isherm = True
        Qobj data =
        [[ 0.  0.  0. ...  0.  0.  0.]
         [ 0.  0. -1. ...  0.  0.  0.]
         [ 0. -1.  0. ...  0.  0.  0.]
         ...
         [ 0.  0.  0. ...  3. -1.  0.]
         [ 0.  0.  0. ... -1.  3.  0.]
         [ 0.  0.  0. ...  0.  0.  4.]]
        """
        if model == "Hubbard":
            H1, H2 = 0, 0
            t, = kwargs.values()

            for idx in track(range(self.sites), description="Hamitonian"):
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
            # Potentially add AIM model
            pass

        return H

    def get_e(self, H: Qobj, states: list) -> float:
        """Outputs a matrix element (energy) from the hamiltonian using
        given kets on which perform projection.

        Parameters
        ----------
        H: Qobj, shape (4^self.sites, 4^self.sites), default=None
            Fermions network hamiltonian.
        states: array-like, shape (2, 1), default=None
            Vectors used to process scalar product on H.

            (ex: states=[bra, ket] as integers to convert from binairy)

        Returns
        -------
        E: int
            Representation of projected vectors on H (energy).

        Examples
        --------
        >>> N = Network(sites_nb=2)
        >>> Hamiltonian = N.get_hamiltonian(model="Hubbard", U=2, t=1)
        >>> N.get_e(H=Hamiltonian, states=[15, 15])
        >>> 2.0
        """
        bra, ket = self.get_state(states[0], type="bra"), self.get_state(states[1])
        E = bra*H*ket

        return E.tr()

    def lanczos(self, H: Qobj, iterations: int, init_state=None) -> tuple:
        """Implementation of Lanczos algorithm for Network hamiltonian.

        Parameters
        ----------
        H: Qobj, shape (4^self.sites, 4^self.sites), default=None
            Fermions network hamiltonian.
        iterations: int, default=None
            Number of iterations on which perform the algorithm.
        init_state: QObj, shape (4^self.sites, 1) default=random
            Initial quantum state to start the first iteration.

        Returns
        -------
        -: tuple, shape (1, 2)
            Respectively the maximum eigenvalue and the associated eigenvector.

        Examples
        --------
        >>> N = Network(sites_nb=2)
        >>> Hamitonian = N.get_hamiltonian(model="Hubbard", U=1, t=1)
        >>> N.lanczos(H=Hamiltonian, iterations=10)
        >>> (2.561552779602264, Quantum object: dims = [[10], [1]], shape = (10, 1), type = ket
        Qobj data =
        [[0.43516215]
         [0.78820544]
         [0.43516215]
         [0.        ]
         [0.        ]
         [0.        ]
         [0.        ]
         [0.        ]
         [0.        ]
         [0.        ]])
        """
        if not init_state:
            dim = H.shape[0]
            state = np.randint(dim - 1)
            init = self.get_state(state=state)
        else:
            init = init_state

        H_n = np.zeros((iterations, iterations), dtype=np.complex64)
        for iter in track(range(iterations), description="Lanczos algorithm"):
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
    N = Network(sites_nb=2)
    H = N.get_hamiltonian(model="Hubbard", U=1, t=1)
    print(N.lanczos(H=H, iterations=10))

