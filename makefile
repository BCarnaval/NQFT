
RM=-rm -f
EX_SRC=./nqft/examples/
GARBAGE={eps,tex,log,aux}

.PHONY: all plot clean

all:

clean:
	@$(RM) $(EX_SRC)*.$(GARBAGE)
	@$(RM) $(EX_SRC)*-inc-*
