
RM=-rm -f
EX_SRC=./nqft/examples/
GARBAGE={eps,tex,log,aux}

.PHONY: all plot clean

all:

plot:
	@gnuplot -p $(EX_SRC)plot_nh_p.gp

clean:
	@$(RM) $(EX_SRC)*.$(GARBAGE)
	@$(RM) $(EX_SRC)*-inc-*
