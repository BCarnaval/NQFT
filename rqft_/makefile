
RM=-rm -f
EX_SRC=./examples/figs/
GARBAGE={eps,tex,log,aux}

.PHONY: all plot clean

all:

plot:
	@gnuplot -p $(EX_SRC)plot_n_nh.gp

clean:
	@$(RM) $(EX_SRC)*.$(GARBAGE)
	@$(RM) $(EX_SRC)*-inc-*
