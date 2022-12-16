
# Global settings
set terminal epslatex standalone
set output 'examples/figs/comp_afm_rust.tex'

set xlabel 'Hole doping $p$'
set ylabel 'Hall number $n_H$'
set key left top
set style line 1 lt 1 pt 4 ps 1 lw 2 lc "#6495ED"
set style line 2 lt 1 pt 8 ps 1 lw 2 lc "#DDA0DD"
set style function lines
set yr [-2:2]
set xzeroaxis lt 1 lc 8
set yzeroaxis lt 1 lc 8

plot './examples/data/merged.dat' u 2:(-$3) with linespoints ls 1 t '$-n_H(p)$ Rust',\
     './examples/data/merged.dat' u 8:(($9 * $9) / $11) with linespoints ls 2 t '$n_H(p)$ afmCond'

unset output

system("pdflatex -output-directory='./examples/figs/' './examples/figs/comp_afm_rust.tex'")
system("make clean")
system("open ./examples/figs/comp_afm_rust.pdf")
system("clear")
