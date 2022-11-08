#!/usr/bin/env bash

INTERACTIONS=$(seq 0.0 0.05 12.0)
PY_PATH=~/.virtualenvs/nqft-uG72oM7R-py3.10/bin/python3
FILE_PATH=./nqft/qcm.py

main () {
    for i in ${INTERACTIONS}; do
        ${PY_PATH} ${FILE_PATH} $i
    done
}

main
