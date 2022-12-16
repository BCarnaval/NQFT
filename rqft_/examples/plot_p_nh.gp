# Global settings
set terminal epslatex standalone
set output 'examples/figs/n_h_doping.tex'

set xlabel '$p$'
set ylabel '$n_H$'
set key at 0.1,1.85
set style line 1 lt 1 pt 4 ps 1 lw 2 lc "#6495ED"
set style function lines
set yr [0:2]
set xr [-0.01:0.35]

plot 'examples/data/conductivities.dat' u 7:3 with linespoints ls 1 t '$n_H(p)$'

unset output

system("pdflatex -output-directory='./examples/figs/' './examples/figs/n_h_doping.tex'")
system("make clean")
system("open ./examples/figs/n_h_doping.pdf")
system("clear")
