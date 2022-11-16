#!/usr/bin/env bash

E_NBR=$(seq 6 2 12)
PY_PATH=~/.virtualenvs/nqft-uG72oM7R-py3.10/bin/python3
FILE_PATH=./nqft/qcm.py

main () {
    for i in ${E_NBR}; do
        ntfy send QcmSimulations "Fermions nb: $i..."
        ${PY_PATH} ${FILE_PATH} $i
    done

    ntfy send QcmSimulations "Simulations done!"
}

main
