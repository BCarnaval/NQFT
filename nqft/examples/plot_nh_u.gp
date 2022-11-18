# Global settings
set terminal epslatex standalone
set output 'nqft/examples/n_h_qcm_u.tex'

# Plot style
set yr [-2:2]
set xlabel '$p$'
set ylabel '$n_H$'
set key samplen 3
set key font ",9"
set key spacing 2.5
set key at -0.55,-0.4
set label "U" at -0.85,-0.25

# Custom linestyles (1, 2)
set style line 1 lt 1 pt 4 ps 1 lw 2 lc "#c9cca1"
set style line 2 lt 1 pt 5 ps 1 lw 2 lc "#caa05a"
set style line 3 lt 1 pt 6 ps 1 lw 2 lc "#ae6a47"
set style line 4 lt 1 pt 7 ps 1 lw 2 lc "#8b4049"
set style line 5 lt 1 pt 8 ps 1 lw 2 lc "#543344"
set style line 6 lt 1 pt 9 ps 1 lw 2 lc "#515262"
set style line 7 lt 1 pt 10 ps 1 lw 2 lc "#63787d"
set style line 8 lt 1 pt 11 ps 1 lw 2 lc "#8ea091"
set style function lines

# Solid at x = y = 0
set xzeroaxis lt 1 lw 1 lc "#000000"
set yzeroaxis lt 1 lw 1 lc "#000000"

plot './nqft/Data/hall_2x2/n_h_2x2_n4_U001_eta01.txt' u 1:2 w lp ls 1 t '2x2, 0.01', \
     './nqft/Data/hall_3x4/n_h_3x4_n12_U001_eta01.txt' u 1:2 w lp ls 6 t '3x4, 0.01', \
     './nqft/Data/hall_2x2/n_h_2x2_n4_U2_eta01.txt' u 1:2 w lp ls 4 t '2x2, 2', \
     './nqft/Data/hall_3x4/n_h_3x4_n12_U2_eta01.txt' u 1:2 w lp ls 5 t '3x4, 2', \

#     './nqft/Data/hall_2x2/n_h_2x2_n4_U05_eta01.txt' u 1:2 w lp ls 2 t '2x2, 0.5', \
#     './nqft/Data/hall_2x2/n_h_2x2_n4_U1_eta01.txt' u 1:2 w lp ls 3 t '2x2, 1', \

#     './nqft/Data/hall_2x2/n_h_2x2_n4_U3_eta01.txt' u 1:2 w lp ls 5 t '2x2, 3', \
#     './nqft/Data/hall_2x2/n_h_2x2_n4_U4_eta01.txt' u 1:2 w lp ls 6 t '2x2, 4', \
#     './nqft/Data/hall_2x2/n_h_2x2_n4_U8_eta01.txt' u 1:2 w lp ls 7 t '2x2, 8', \

unset output

system("pdflatex -output-directory='./nqft/examples/' './nqft/examples/n_h_qcm_u.tex'")
system("make clean")
system("open ./nqft/examples/n_h_qcm_u.pdf")
system("clear")
