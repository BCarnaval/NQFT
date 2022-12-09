# Global settings
set terminal epslatex standalone
set output 'nqft/examples/figures/article.tex'

# Plot style
set yr [-2:2]
set xlabel '$p$'
set ylabel '$n_H$'
set key samplen 3
set key font ",9"
set key spacing 2.5
set key at -0.45,-1.2
set label "$(t', t'')$" at -0.8,-1

# Custom linestyles (1, 2)
set style line 3 lt 1 pt 6 ps 1 lw 2 lc "#ae6a47"
set style line 4 lt 1 pt 7 ps 1 lw 2 lc "#8b4049"
set style line 5 lt 1 pt 8 ps 1 lw 2 lc "#543344"
set style function lines

# Solid at x = y = 0
set xzeroaxis lt 1 lw 1 lc "#000000"
set yzeroaxis lt 1 lw 1 lc "#000000"

plot './nqft/Data/data_article/nh_t.txt' u 1:2 w lp ls 3 t '$(0.0, 0.0)t$', \
     './nqft/Data/data_article/nh_t_tp.txt' u 1:2 w lp ls 4 t '$(-0.3, 0.0)t$', \
     './nqft/Data/data_article/nh_t_tp_tpp.txt' u 1:2 w lp ls 5 t '$(-0.3, 0.2)t$', \


unset output

system("pdflatex -output-directory='./nqft/examples/figures/' './nqft/examples/figures/article.tex'")
system("make clean")
system("open ./nqft/examples/figures/article.pdf")
system("clear")
