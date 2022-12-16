# Global settings
set terminal epslatex standalone
set output 'examples/figs/n_h_doping_peter_36.tex'

set xlabel '$p$'
set ylabel '$n_H$'
set yr [0:1.8]
set xr [-0.01:0.35]
set key at 0.1, 1.73
set style line 1 lt 1 pt 4 ps 1 lw 2 lc "#ae6a47"
set style function lines
set grid

plot 'examples/data/conductivities.dat' u 7:3 with linespoints ls 1 t '$n_H(p)$'

unset output

system("pdflatex -output-directory='./examples/figs/' './examples/figs/n_h_doping_peter_36.tex'")
system("make clean")
system("open ./examples/figs/n_h_doping_peter_36.pdf")
system("clear")
