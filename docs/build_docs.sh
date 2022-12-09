#!/usr/bin/env bash

PROJECT=nqft
MAKE_DIR=./docs
DOCS_SRC=./docs/source
HTML_SRC=./docs/build/html
LATEX_SRC=./docs/build/latex

clean_html () {
    make --directory=${MAKE_DIR} clean
}

build_doc () {
    sphinx-apidoc -f -o ${DOCS_SRC} ../nqft/

    if [[ "${1}" == "html" ]]; then
        OPT='html'
    elif [[ "${1}" == "pdf" ]]; then
        OPT='latexpdf'
    fi

    make --directory=${MAKE_DIR} ${OPT}
}

display_doc () {
    if [[ "${1}" == "html" ]]; then
        FILE=${HTML_SRC}/${PROJECT}.html
    elif [[ "${1}" == "pdf" ]]; then
        FILE=${LATEX_SRC}/${PROJECT}.pdf
    fi

    open ${FILE}
}

main () {
    clean_html
    build_doc $1
    display_doc $1
}

main "${@}"
