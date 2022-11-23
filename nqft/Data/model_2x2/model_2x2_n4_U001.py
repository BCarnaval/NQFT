from pyqcm import *
new_cluster_model('clus', 4, 0, generators=None, bath_irrep=False)
add_cluster('clus', [0, 0, 0], [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], ref = 0)
lattice_model('model_2x2_n4_U001', [[2, 0, 0], [0, 2, 0]], None)
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
set_target_sectors(['R0:N4:S0'])
set_parameters("""

                U = 0.01
                t = 1.0
                tp = -0.3
                tpp = 0.2
                mu = 4.0
                """)
set_parameter("U", 0.01)
set_parameter("mu", 4.0)
set_parameter("t", 1.0)
set_parameter("tp", -0.3)
set_parameter("tpp", 0.2)

new_model_instance(0)

solution=[None]*1

#--------------------- cluster no 1 -----------------
solution[0] = """
U	0.01
mu	4
t	1
tp	-0.3

GS_energy: -19.9925 GS_sector: R0:N4:S0:1
GF_format: bl
mixing	0
state
R0:N4:S0	-19.9925	1
w	4	22
-4.2975040057382	0.0026189160801913	-0.4999927472147	0.4999927472147	-0.0026189160801909
-4.2975040057382	-0.49999274724611	-0.0026189160803553	0.0026189160803559	0.49999274724611
-5.6950054628979	0.49999968334864	0.49999968334862	0.49999968334862	0.49999968334864
-5.6999981306572	0.00054059002073917	-0.00054059000205145	-0.00054059000205006	0.00054059002073793
-7.0975007819575	2.5822192468186e-06	-0.00044720895202165	0.0004472089520227	-2.5822192463023e-06
-7.0975007819575	-0.00044720895205066	-2.5822192466503e-06	2.5822192468002e-06	0.00044720895205078
-8.3000094872506	0.00028588813956359	0.00031160021415359	-0.00031160021415256	-0.00028588813956248
-8.3000094872506	-0.00031160021419495	0.00028588813960121	-0.00028588813960162	0.00031160021419487
-8.3000211948219	-9.0400416299492e-05	8.3630634328597e-05	-8.3630634328572e-05	9.0400416299208e-05
-8.3000211948219	8.3630634459417e-05	9.0400416441053e-05	-9.0400416441116e-05	-8.3630634459993e-05
-9.6950148379015	0.00015624878761666	-0.00015624878761653	-0.00015624878761651	0.00015624878761653
-4.2924949634021	-0.49999287908196	-0.0026207895793654	0.0026207895793657	0.49999287908196
-4.2924949634021	-0.0026207895795301	0.49999287911337	-0.49999287911337	0.0026207895795304
-1.6949945371021	0.49999968334862	-0.49999968334864	-0.49999968334864	0.49999968334862
-1.689995891815	0.00054059000275734	0.00054059002003123	0.00054059002003128	0.00054059000275879
-0.29000096078341	-0.00024632490822395	0.00018163503356422	-0.0001816350335648	0.00024632490822419
-0.29000096078341	0.00018163503369261	0.00024632490840103	-0.00024632490839821	-0.0001816350336953
-0.28998919043578	-0.00018823054767072	-0.00025633835647233	0.00025633835647279	0.00018823054767053
-0.28998919043578	-0.00025633835664007	0.00018823054779325	-0.00018823054779512	0.00025633835664159
0.90751955481754	-0.000239803877291	0	0	0.00023980387729005
0.90751955481754	0	-0.00023980387730493	0.00023980387730528	0
2.3050148379015	-0.00015624878761731	-0.00015624878761588	-0.00015624878761561	-0.00015624878761744

"""
read_cluster_model_instance(solution[0], 0)