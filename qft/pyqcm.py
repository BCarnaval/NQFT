from pyqcm import *
from pyqcm.draw_operator import *


new_cluster_model('clus', 4)
add_cluster('clus', [0,0,0], [[0,0,0], [1,0,0], [2,0,0],[3,0,0]])
lattice_model('1D_4', [[4,0,0]])
interaction_operator('U')
hopping_operator('t', [1,0,0], -1)

draw_operator('U')
