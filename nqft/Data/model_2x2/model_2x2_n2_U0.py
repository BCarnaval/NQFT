from pyqcm import *
new_cluster_model('clus', 4, 0, generators=None, bath_irrep=False)
add_cluster('clus', [0, 0, 0], [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], ref = 0)
lattice_model('model_2x2_n2_U0', [[2, 0, 0], [0, 2, 0]], None)
interaction_operator('U', band1=1, band2=1)
hopping_operator('t', [1, 0, 0], -1, band1=1, band2=1)
hopping_operator('t', [0, 1, 0], -1, band1=1, band2=1)
hopping_operator('tp', [1, 1, 0], -1, band1=1, band2=1)
hopping_operator('tp', [-1, 1, 0], -1, band1=1, band2=1)
hopping_operator('tpp', [2, 0, 0], -1, band1=1, band2=1)
hopping_operator('tpp', [0, 2, 0], -1, band1=1, band2=1)

try:
    import model_extra
except:
    pass
set_target_sectors(['R0:N2:S0'])
set_parameters("""

                U = 0.0
                t = -1.0
                tp = 0.3
                tpp = -0.2
                """)
set_parameter("U", 0.0)
set_parameter("t", -1.0)
set_parameter("tp", 0.3)
set_parameter("tpp", -0.2)

new_model_instance(0)

solution=[None]*1

#--------------------- cluster no 1 -----------------
solution[0] = """
t	-1
tp	0.3

GS_energy: -4.6 GS_sector: uncorrelated
GF_format: bl
mixing	0
state
R0	-4.6	1
w	4	4
-2.3	0.5	-0.5	-0.5	0.5
0.3	-0.68586745861084	-0.17200531741405	0.17200531741405	0.68586745861084
0.3	-0.17200531741405	0.68586745861084	-0.68586745861084	0.17200531741405
1.7	0.5	0.5	0.5	0.5

"""
read_cluster_model_instance(solution[0], 0)
