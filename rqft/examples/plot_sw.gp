# Global settings
set output 'examples/figs/s_w.pdf'

set xrange[-pi:pi]
set yrange[-pi:pi]
unset key
set view map
set tics out
set tics nomirror

set pm3d map
set palette defined (-10 "#FFFFFF", 0 "#6495ED", 10 "#708090")

splot 'examples/data/spectral.dat'
