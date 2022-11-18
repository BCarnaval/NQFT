# Global settings
set terminal epslatex standalone
set output 'nqft/examples/n_h_qcm.tex'

# Plot style
set yr [-2:2]
set xlabel '$p$'
set ylabel '$n_H$'
set key at -0.1,1.8

# Custom linestyles (1, 2)
set style line 1 lt 1 pt 4 ps 1 lw 2 lc "#6495ED"
set style line 2 lt 1 pt 8 ps 1 lw 2 lc "#ffb6c1"
set style function lines

# Solid at x = y = 0
set xzeroaxis lt 1 lw 1 lc "#000000"
set yzeroaxis lt 1 lw 1 lc "#000000"

plot 'nqft/examples/n_h_article.dat' u 2:3 w lp ls 1 t '$n_H(p)$ (article)', \
     './nqft/Data/n_h_3x4_n12_U001_eta01.txt' u 1:2 w lp ls 2 t '$n_H(p)$ (pyqcm)'

unset output

system("pdflatex -output-directory='./nqft/examples/' './nqft/examples/n_h_qcm.tex'")
system("make clean")
system("open ./nqft/examples/n_h_qcm.pdf")
system("clear")
