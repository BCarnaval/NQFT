"""This module contains global functions used in the 
expression and operation on Hubbard hamiltonian and 
other Fock space operations.
"""

from qutip import Qobj


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

