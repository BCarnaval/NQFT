"""Hall effect in cuprates with an incommensurate collinear
spin-density wave.
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.constants import e, pi


def build_dispersion(hop_amps: list, mu: float, lims: tuple,
        derivative=(None, 0), res=100) -> None:
    """Build 3D dispersion relation grid and outputs it as 2D arrays.

    Parameters
    ----------
    hop_amps: list, size=3, default=None
        Hopping coefficients module (t, t', t'').
    mu: float, default=None
        NA.
    lims: tuple, size=2, default=None
        Wavevectors limit values.
    derivative: tuple, size=2, default=(None, 0)
        Specifies derivation variable &  order of derivation.
    res: int, default=100
        Resolution of outputed grid.

    Returns
    -------
    X, Y, Z: tuple
        2D np.ndarrays representing a grid for kx, a grid for ky and
        a grid for eigenenergies.
    """
    variable, order = derivative
    t, t1, t2 = hop_amps
    kx = np.arange(lims[0], lims[1], 1/res)
    ky = kx
    X, Y = np.meshgrid(kx, ky)

    # Computing z component of dispertion relation
    if not variable or order > 2:
        first = -2*t*(np.cos(X) + np.cos(Y))
        second = -2*t1*(np.cos(X + Y) + np.cos(X - Y))
        third = -2*t2*(np.cos(2*X) + np.cos(2*Y))
        if order > 2:
            print(f"Order {order} in derivative isn't defined within "
                   "this function.")
        else:
            pass

    elif variable == 'kx':
        mu = 0 # Setting mu to 0 because it doesn't appear in any derivative
        if order == 1:
            first = 2*t*np.sin(X)
            second = 2*t1*(np.sin(X + Y) + np.sin(X - Y))
            third = 4*t2*np.sin(2*X)

        elif order == 2:
            first = 2*t*np.cos(X)
            second = 2*t1*(np.cos(X + Y) + np.cos(X - Y))
            third = 8*t2*np.cos(2*X)

    elif variable == 'ky':
        if order == 1:
            first = 2*t*np.sin(Y)
            second = 2*t1*(np.sin(X + Y) - np.sin(X - Y))
            third = 4*t2*np.sin(2*Y)

        elif order == 2:
            first = 2*t*np.cos(Y)
            second = 2*t1*(np.cos(X + Y) + np.cos(X - Y))
            third = 8*t2*np.cos(2*Y)

    Z = first + second + third - mu

    return X, Y, Z

def spectral_weight(omega: float, eta: float, coords: tuple,
                    show=False) -> None:
    """Outputs the spectral_weight as a 2D numpy array.

    Parameters
    ----------
    omega: float, default=None
        Frequency of the system.
    eta: float, default=None
        The lorentzian broadening module.
    coords: tuple, size=3, default=None
        3D grid of the eigenenergies as a function of wavevectors (kx, ky).
    show: bool, default=False
        If True, outputs a colormesh of the spectral weight.

    Returns
    -------
    A: np.ndarray, shape (2*lims[0]*res, 2*lims[1]*res)
        Spectral weight at frequency 'omega'.

    Examples
    --------
    >>> E = build_dispersion([1, 0, 0], 0.0, (-np.pi, np.pi), res=10)
    >>> A = spectral_weight(0.0, 0.05, E)
    >>> [[0.00099456 0.00099955 0.00101469 ... 0.00103539 0.00101141 0.00099801]
    [0.00099955 0.00100457 0.00101982 ... 0.00104068 0.00101653 0.00100302]
    [0.00101469 0.00101982 0.00103542 ... 0.00105677 0.00103205 0.00101824]
                                      ...
    [0.00103539 0.00104068 0.00105677 ... 0.00107877 0.00105329 0.00103905]
    [0.00101141 0.00101653 0.00103205 ... 0.00105329 0.00102869 0.00101495]
    [0.00099801 0.00100302 0.00101824 ... 0.00103905 0.00101495 0.00100148]]
    """
    X, Y, E = coords
    A = 1/pi*(eta/((omega - E)**2 + eta**2))

    if show:
        fig = plt.figure()
        ax = fig.add_subplot()
        spectral = ax.pcolormesh(X, Y, A, cmap=cm.Blues)

        # Graph format & style
        ax.set_xlabel("$k_x$")
        ax.set_ylabel("$k_y$")

        ax.set_xticks(ticks=[-pi, 0, pi], labels=["$-\pi$", "0", "$\pi$"])
        ax.set_yticks(ticks=[-pi, 0, pi], labels=["$-\pi$", "0", "$\pi$"])

        fig.colorbar(spectral)
        plt.show()
    else:
        pass

    return A

def sigma_xx(V: float) -> float:
    """Computing longitudinal conductivity at zero temperature
    in the zero-frequency limit when interband transitions can be
    neglected.
    """
    coeff = e**2*pi/V

    return

def sigma_xy(V: float) -> float:
    """Computing transversal conductivity at zero temperature
    in the zero-frequency limit when interband transitions can be
    neglected.
    """
    coeff = e**3*pi**2/(3*V)

    return


if __name__ == "__main__":
    X, Y, E = build_dispersion(hop_amps=[1, 0.0, 0.0], mu=0.0, lims=(-pi,
        pi), derivative=('kx', 3), res=100)

    A = spectral_weight(omega=0, eta=0.05, coords=(X, Y, E), show=True)

