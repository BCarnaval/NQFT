from qft import __version__
from qft.hamiltonian import funcs

def test_version():
    assert __version__ == '0.1.0'

def test_hamiltonian():
    for site in range(2, 9):
        network = funcs.Network(sites_nb=site)
        H = network.get_hamiltonian(model="Hubbard", U=1, t=1)
        assert H.shape == (4**site, 4**site)

