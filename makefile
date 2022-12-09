
RM=-rm -f
DOCS_SCRIPT=./docs/build_docs.sh
EX_SRC=./nqft/examples/figures/
GARBAGE={eps,tex,log,aux}

.PHONY: all clean pdf html

all:

pdf:
	$(DOCS_SCRIPT) pdf

html:
	$(DOCS_SCRIPT) html

clean:
	@$(RM) $(EX_SRC)*.$(GARBAGE)
	@$(RM) $(EX_SRC)*-inc-*
