#!/usr/bin/env python3

"""Module de test des fonctionnalités de la librairie
Python Qutip. 
"""

from qutip import (basis, create, destroy, num, tensor, identity)

vaccum = basis(2)
creation = create(2)
anihilation = destroy(2)
number = num(2)
I = identity(2)

def get_state(state: int, type="ket"):
    """Gives array-like representation of given state using 
    Qutip 'tensor' function. 
    """
    binairy = format(state, 'b').zfill(4)
    ket = [*binairy]

    vec_state = []
    for presence in ket:
        if int(presence):
            vec_state.append(creation*vaccum)
        else:
            vec_state.append(vaccum)

    if type == "bra":
        return tensor(*vec_state).dag()
    elif type == "ket":
        return tensor(*vec_state)

def get_H(model: str, U: int, **kwargs):
    """Outputs the hamiltonian of specified many-body model.
    """
    if model == "Hubbard":
        t, = kwargs.values()

        H1 = tensor(number, I, number, I) + tensor(I, number, I, number)

        H2 = tensor(anihilation, creation, I, I) + tensor(creation,
                anihilation, I, I) + tensor(I, I, creation, anihilation) + tensor(I, I, anihilation, creation)

        H = U*H1 - t*H2

    elif model == "AIM":
        mu, theta, e = kwargs.values()

        H1 = tensor(I, number, I, number)

        H2 = tensor(I, number, I, I) + tensor(I, I, I, number)

        H3 = tensor(creation, anihilation, I, I) + tensor(anihilation,
                creation, I, I) + tensor(I, I, creation, anihilation) + tensor(I, I, anihilation, creation)

        H4 = tensor(number, I, I, I) + tensor(I, I, number, I)

        H = U*H1 - mu*H2 - theta*H3 + (e - mu)*H4

    return H

def get_E(model: str, states: list, U: int, **kwargs):
    """Outputs a matrix element (energy) from the hamiltonian using 
    given kets on which perform projection.
    """
    bra, ket = get_state(states[0], type="bra"), get_state(states[1])
    H = get_H(model=model, U=U, **kwargs)

    return bra*H*ket


if __name__ == "__main__":
    print(get_H(model="Hubbard", U=1, t=1))
    # print(get_H(model="AIM", U=1, mu=1, theta=1, e=1))
    # print(get_E(model="AIM", states=[15, 15], U=4, mu=10, theta=1, e=1))
