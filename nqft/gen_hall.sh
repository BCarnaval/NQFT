#!/usr/bin/env bash

MUS=$(seq -4.0 0.1 4.0)
PY_PATH=~/.virtualenvs/nqft-uG72oM7R-py3.10/bin/python3
FILE_PATH=./nqft/qcm.py

main () {
    for i in ${MUS}; do
        ntfy send QcmSimulations "Chemical potential: $i..."
        ${PY_PATH} ${FILE_PATH} $i 12
    done

    ntfy send QcmSimulations "Simulations done!"
}

main
