"""Hall effect in cuprates with an incommensurate collinear
spin-density wave.
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


def dispersion(hop_amp: list, mu: float, lims: tuple, res=100) -> None:
    """Docs
    """
    t, t1, t2 = hop_amp
    kx = np.arange(lims[0], lims[1], 1/res)
    ky = kx
    X, Y = np.meshgrid(kx, ky)

    # Computing z component of dispertion relation
    first = -2*t*(np.cos(X) + np.cos(Y))
    second = -2*t1*(np.cos(X + Y) + np.cos(X - Y))
    third = -2*t2*(np.cos(2*X) + np.cos(2*Y))
    Z = first + second + third - mu

    # Access the z=0 contour line of the fermi surface
    fermi_surface = plt.contourf(X, Y, Z, levels=0, cmap=cm.GnBu).collections
    x, y = fermi_surface[0].get_paths()[0].vertices.T


    return


if __name__ == "__main__":
    X, Y, Z = dispersion(hop_amp=[1, -0.4, 0.1], mu=0.9, lims=(-np.pi, np.pi),
            res=100)
