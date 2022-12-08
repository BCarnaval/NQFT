"""This module contains global functions used in the
expression and operation on Hubbard hamiltonian.

It is also used more globally to perform operations on data files
containing spectral function.
"""

import numpy as np
import pandas as pd
from rich import print
from qutip import Qobj
from colour import Color
from scipy.constants import pi
from matplotlib.colors import LinearSegmentedColormap


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

    Examples
    --------
    >>> delta(1, 1)
    1.0
    >>> delta(0, 1)
    0.0
    """
    return 1.0 if j == k else 0.0


def read_fermi_arc(path="./nqft/Data/fermi_arc_data", size=36,
                   res=200) -> dict:
    """Reads Peter's data on spectral weight at Fermi
    level for a given number of sites.

    Parameters
    ----------
    path: str, default="./nqft/Data/fermi_arc_data/"
        Path to data directory.

    size: int, default=36
        Number of sites of studied model.

    res: int, default=200
        Resolution of momentum space.

    Returns
    -------
    arcs: dict, size=6
        A dict containing all spectral functions.
    """
    if size == 36:
        size_path = "_N36/"
        files = [
            f"{path}{size_path}Akw_N24.npy",
            f"{path}{size_path}Akw_N26.npy",
            f"{path}{size_path}Akw_N28.npy",
            f"{path}{size_path}Akw_N30.npy",
            f"{path}{size_path}Akw_N32.npy",
            f"{path}{size_path}Akw_N34.npy",
            f"{path}{size_path}Akw_N36.npy",
        ]

        extensions = ["N24", "N26", "N28", "N30", "N32", "N34", "N36"]

    elif size == 64:

        if res == 200:
            size_path = "_N64/nk_200/"

        elif res == 500:
            size_path = "_N64/nk_500/"

        files = [
            f"{path}{size_path}Akw_N48.npy",
            f"{path}{size_path}Akw_N52.npy",
            f"{path}{size_path}Akw_N56.npy",
            f"{path}{size_path}Akw_N60.npy",
            f"{path}{size_path}Akw_N64.npy",
        ]

        extensions = ["N48", "N52", "N56", "N60", "N64"]

    arcs = {
        ext: 1 / pi * np.load(file) for ext, file in zip(extensions, files)
    }

    return arcs


def flatten_fermi_arc(save_path="./nqft/Data/fermi_arc_data_1D", size=36,
                      res=200, type="text") -> None:
    """Saves Peter's Fermi arc numpy 2D arrays files as 1D
    arrays numpy files so they can be read in Rust/C.

    Parameters
    ----------
    save_path: str, default="./nqft/Data/fermi_arc_data_1D/"
        Path where to save flatten arrays.

    size: int, default=36
        Number of sites of studied model.

    res: int, default=200
        Resolution of momentum space.

    type: str, default="text"
        Format in which save flattened arrays
        (ex: type="npy" leads to .npy files)

    Returns
    -------
    None
    """
    if size == 36:
        save_path += "_N36/"

    elif size == 64:
        save_path += "_N64/"

        if res == 200:
            save_path += "nk_200/"

        elif res == 500:
            save_path += "nk_500/"

    arcs_files = read_fermi_arc(size=size, res=res)

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


def find_nearest(array, value) -> int:
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

    Examples
    --------
    >>> a = np.array([0, 1, 2, 3, 4, 5, 6])
    >>> find_nearest(a, 3)
    3
    >>> b = np.array([0.0, 1.111, 2.5, 2.6])
    >>> find_nearest(b, 2.0)
    2
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def add_column(file: str, column: np.array, idx: int) -> None:
    """Insert new data to given text file as a column.

    Parameters
    ----------
    file: str, default=None
        File in which add column.

    column: np.array, default=None
        Data column to add.

    idx: int, default=0
        Index of the new column inside current file.
    """
    file_to_df = pd.read_csv(file, delimiter=' ', header=None)

    if len(file_to_df[0]) != len(column):
        print(f"Data column must have same dimension as: {file}")
        print(f"Found dimension of: {len(column)} and {len(file_to_df)}.")
        return

    else:
        file_to_df.insert(idx, column)
        file_to_df.to_csv(file, sep=' ', header=False, index=False)

    return


def make_cmap(ramp_colors: list) -> LinearSegmentedColormap:
    """Makes a custom colormap to use in matplotlib 'contourf' or any plot
    using a colorbar.

    Parameters
    ----------
    ramp_colors: list, default=None
        Hex code(s) of colors to use when making the colormap.

    Returns
    -------
    color_ramp: LinearSegmentedColormap
        Custom colormap
    """
    color_ramp = LinearSegmentedColormap.from_list(
        'my_list', [Color(c1).rgb for c1 in ramp_colors])
    return color_ramp


if __name__ == "__main__":
    pass
