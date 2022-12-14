# Global settings
set terminal epslatex standalone
set output 'nqft/examples/figures/n_h_qcm.tex'

# Plot style
set yr [-2:2]
set xlabel '$p$'
set ylabel '$n_H$'
set key at -0.35,1.8

# Custom linestyles (1, 2)
set style line 1 lt 1 pt 4 ps 1 lw 2 lc "#ae6a47"
set style line 2 lt 1 pt 8 ps 1 lw 2 lc "#543344"
set style function lines

# Solid at x = y = 0
set xzeroaxis lt 1 lw 1 lc "#000000"
set yzeroaxis lt 1 lw 1 lc "#000000"

plot 'nqft/Data/data_article/nh_t_tp_tpp.txt' u 1:2 w lp ls 1 t '$U=0.0$', \
     './nqft/Data/hall_3x4/n_h_3x4_n12_U001_eta01.txt' u 1:2 w lp ls 2 t '$U=0.01$'

unset output

system("pdflatex -output-directory='./nqft/examples/figures/' './nqft/examples/figures/n_h_qcm.tex'")
system("make clean")
system("open ./nqft/examples/figures/n_h_qcm.pdf")
system("clear")
