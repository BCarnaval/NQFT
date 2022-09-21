from qft import __version__
from qft.hamiltonian import qutip_funcs

def test_version():
    assert __version__ == '0.1.0'

def test_hamiltonian():
    H = qutip_funcs.get_hamiltonian(model="Hubbard", U=1, t=1)
    assert H.shape == 16

