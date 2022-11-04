from pyqcm import *
new_cluster_model('clus', 16, 0, generators=None, bath_irrep=False)
add_cluster('clus', [0, 0, 0], [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0], [0, 2, 0], [1, 2, 0], [2, 2, 0], [3, 2, 0], [0, 3, 0], [1, 3, 0], [2, 3, 0], [3, 3, 0]], ref = 0)
lattice_model('model_4x4_n10_U0', [[4, 0, 0], [0, 4, 0]], None)
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
set_target_sectors(['R0:N10:S0'])
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
tpp	-0.2

GS_energy: -24.717 GS_sector: uncorrelated
GF_format: bl
mixing	0
state
R0	-24.717	1
w	16	16
-3.6691026902267	0.1250347043319	-0.21639347133636	0.21639347133636	-0.1250347043319	-0.21639347133636	0.37511871965903	-0.37511871965903	0.21639347133636	0.21639347133636	-0.37511871965903	0.37511871965903	-0.21639347133636	-0.1250347043319	0.21639347133636	-0.21639347133636	0.12503470433191
-2.5417265989197	-0.29087492983566	0.36355619328299	-0.18779430368754	0.021033957494676	0.34007574184302	-0.33544674860884	0.024257066966727	0.1369127642448	-0.1369127642448	-0.024257066966727	0.33544674860884	-0.34007574184302	-0.021033957494676	0.18779430368754	-0.36355619328299	0.29087492983565
-2.5417265989197	0.021033957494676	0.1369127642448	-0.34007574184302	0.29087492983566	-0.18779430368754	0.024257066966727	0.33544674860884	-0.36355619328299	0.36355619328299	-0.33544674860884	-0.024257066966727	0.18779430368754	-0.29087492983566	0.34007574184302	-0.1369127642448	-0.021033957494676
-1.7137439085024	-0.34837003288583	0.22978509993363	0.22978509993363	-0.34837003288583	0.22978509993363	-0.15177593974048	-0.15177593974048	0.22978509993363	0.22978509993364	-0.15177593974048	-0.15177593974048	0.22978509993364	-0.34837003288583	0.22978509993364	0.22978509993364	-0.34837003288583
-0.70548171417292	0.28965191582535	-0.186068993535	0.186068993535	-0.28965191582535	-0.18606899353501	-0.31122086522026	0.31122086522026	0.18606899353501	0.18606899353501	0.31122086522026	-0.31122086522026	-0.18606899353501	-0.28965191582535	0.186068993535	-0.186068993535	0.28965191582535
-0.7	0	0.35355339059328	-0.35355339059328	0	-0.35355339059327	0	0	0.35355339059327	0.35355339059327	0	0	-0.35355339059327	0	-0.35355339059328	0.35355339059328	0
-0.24336245325013	-0.51015947214215	0.033659491117329	0.31643772719503	-0.008723643424054	0.02285394927279	0.19580532154585	0.003348238931105	-0.31547135661965	0.31547135661965	-0.0033482389311045	-0.19580532154585	-0.02285394927279	0.0087236434240538	-0.31643772719503	-0.033659491117329	0.51015947214215
-0.24336245325013	-0.0087236434240539	-0.31547135661965	-0.022853949272789	0.51015947214215	0.31643772719503	0.0033482389311047	-0.19580532154585	-0.033659491117329	0.033659491117328	0.19580532154585	-0.0033482389311047	-0.31643772719503	-0.51015947214215	0.02285394927279	0.31547135661965	0.008723643424054
0.77458440439959	-0.38790216597918	-0.20869162706699	0.20869162706699	0.38790216597918	-0.20869162706699	-0.11147878368965	0.11147878368965	0.20869162706699	0.20869162706699	0.11147878368965	-0.11147878368965	-0.20869162706699	0.38790216597918	0.20869162706699	-0.20869162706699	-0.38790216597918
1.1354122485737	-0.033922958116716	0.30561792453518	-0.10902705933978	0.16048370706776	-0.2859581357223	0.11055642078059	-0.52302349889319	0.016019964321585	-0.016019964321585	0.52302349889319	-0.11055642078059	0.2859581357223	-0.16048370706776	0.10902705933978	-0.30561792453518	0.033922958116716
1.1354122485737	0.16048370706776	0.016019964321584	0.2859581357223	0.033922958116716	-0.10902705933978	-0.52302349889319	-0.11055642078059	-0.30561792453518	0.30561792453518	0.11055642078059	0.52302349889319	0.10902705933979	-0.033922958116716	-0.2859581357223	-0.016019964321585	-0.16048370706776
1.2947573487946	0.31852638123106	0.12944436810633	0.12944436810633	0.31852638123106	0.12944436810634	-0.33915963142884	-0.33915963142884	0.12944436810634	0.12944436810634	-0.33915963142885	-0.33915963142884	0.12944436810634	0.31852638123106	0.12944436810633	0.12944436810633	0.31852638123106
1.3	0	-0.35355339059327	-0.35355339059328	0	0.35355339059327	0	0	0.35355339059327	0.35355339059327	0	0	0.35355339059327	0	-0.35355339059327	-0.35355339059327	0
1.9496768035961	-0.00011607329970968	-0.17968897763756	-0.35008746486693	-0.35735802503884	0.17946159202374	-8.1371003064117e-05	-0.2505191204455	-0.34997080933161	0.34997080933161	0.2505191204455	8.1371003064162e-05	-0.17946159202374	0.35735802503884	0.35008746486693	0.17968897763756	0.00011607329970984
1.9496768035961	-0.35735802503884	-0.34997080933161	-0.17946159202374	0.00011607329970953	-0.35008746486693	-0.2505191204455	8.1371003064521e-05	0.17968897763756	-0.17968897763756	-8.1371003064135e-05	0.2505191204455	0.35008746486693	-0.00011607329970979	0.17946159202374	0.34997080933161	0.35735802503884
2.8189865597078	-0.16486134976693	-0.23546329525861	-0.23546329525861	-0.16486134976693	-0.23546329525861	-0.33456659804132	-0.33456659804132	-0.23546329525861	-0.23546329525861	-0.33456659804132	-0.33456659804132	-0.23546329525861	-0.16486134976693	-0.23546329525861	-0.23546329525861	-0.16486134976693

"""
read_cluster_model_instance(solution[0], 0)
