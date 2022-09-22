from qft import __version__
from qft.hamiltonian import funcs

def test_version():
    assert __version__ == '0.1.0'

def test_hamiltonian():
    sites = 4
    network = funcs.Network(sites_nb=sites)
    H = network.get_hamiltonian(model="Hubbard", U=1, t=1)

    assert H.shape == (4**sites, 4**sites)

