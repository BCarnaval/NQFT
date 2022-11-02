"""This module contains global functions used in the
expression and operation on Hubbard hamiltonian and
other Fock space operations.
"""

import numpy as np
from glob import glob
from rich import print
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
        val = m.dag() * n
    else:
        val = m.dag() * m
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


def read_fermi_arc(path="./nqft/Data/fermi_arc_data/") -> dict:
    """Reads Peter's data on spectral weight at Fermi
    level for a given number of sites.

    Parameters
    ----------
    path: str, default="./nqft/Data/fermi_arc_data/"
        Path to data directory.

    Returns
    -------
    arcs: dict, size=6
        A dict containing all spectral functions.
    """
    files = [
        f"{path}Akw_N24.npy",
        f"{path}Akw_N28.npy",
        f"{path}Akw_N30.npy",
        f"{path}Akw_N32.npy",
        f"{path}Akw_N36.npy",
    ]

    extensions = ["N24", "N28", "N30", "N32", "N36"]
    arcs = {ext: np.load(file) for ext, file in zip(extensions, files)}

    return arcs


def read_locals(shape: tuple[int], interaction: float):
    """Reads local model's spectral functions for given shape
    and electronic density.

    Parameters
    ----------
    shape: tuple, size=2, default=None
        Shape of the clusters.
    filling: int, default=None
        Number of electrons in each cluster.

    Returns
    -------


    Examples
    --------
    """
    U_f_to_str = str(interaction).split('.')
    U_str = "".join(U_f_to_str if U_f_to_str[-1] != '0' else U_f_to_str[:-1])

    path = f"./nqft/Data/model_{shape[0]}x{shape[1]}/spectrums"
    files = glob(f"{path}/spectrum_n*_U{U_str}.npy")

    if not files:
        print(f"No model inside: {path}")
        exit(0)
    else:
        pass

    density = [n.split("_n")[-1].split("_")[0] for n in files]
    spectrums = {n: np.load(file) for n, file in zip(density, files)}

    return spectrums


def flatten_fermi_arc(save_path="./nqft/Data/fermi_arc_data_1D/",
                      type="text") -> None:
    """Saves Peter's Fermi arc numpy 2D arrays files as 1D
    arrays numpy files so they can be read in Rust/C.

    Parameters
    ----------
    save_path: str, default="./nqft/Data/fermi_arc_data_1D/"
        Path where to save flatten arrays.

    type: str, default="text"
        Format in which save flattened arrays
        (ex: type="npy" leads to .npy files)

    Returns
    -------
    None
    """
    arcs_files = read_fermi_arc()
    for (ext, array) in arcs_files.items():
        flat_array = np.ravel(array)
        if type == "text":
            np.savetxt(f'{save_path}Akw_{ext}.csv', flat_array)
        elif type == "npy":
            np.save(f'{save_path}Akw_{ext}', flat_array)
        else:
            print("No valid type provided. "
                  "Please select between ('npy' and 'text')")
    return


def find_nearest(array, value):
    """Finds index in given array of the closest value
    of parameter 'value'.

    Parameters
    ----------
    array: array-like, shape=(m, n), default=None
        Array in which search for the value.

    value: int, float, default=None
        Value to search for in array.

    Returns
    -------
    idx: int
        Index at which user can find the closest value in array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


if __name__ == "__main__":
    flatten_fermi_arc()
