# Global settings
set terminal epslatex standalone
set output 'nqft/examples/figures/n_h_filter.tex'

# Plot style
set yr [-2:2]
set xlabel '$p$'
set ylabel '$n_H$'
set key samplen 3
set key font ",9"
set key spacing 2.5
set key at -0.3,-1.25

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

plot './nqft/Data/data_filter/filtered.txt' u 1:2 w lp lt 1 pt 6 ps 1 lw 2 lc "#f0e2db" notitle, \
     './nqft/Data/data_filter/normal.txt' u 1:2 w lp lt 1 pt 6 ps 1 lw 2 lc "#efdcde" notitle, \
     './nqft/Data/data_filter/filtered_2.txt' u 1:2 w lp ls 3 t 'Filtered', \
     './nqft/Data/data_filter/normal_2.txt' u 1:2 w lp ls 4 t 'Not filtered'


unset output

system("pdflatex -output-directory='./nqft/examples/figures/' './nqft/examples/figures/n_h_filter.tex'")
system("make clean")
system("open ./nqft/examples/figures/n_h_filter.pdf")
system("clear")
