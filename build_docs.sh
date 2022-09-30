#!/usr/bin/env bash

PROJECT=nqft
MAKE_DIR=./docs
DOCS_SRC=./docs/source
HTML_SRC=./docs/build/html

clean_html () {
    make --directory=${MAKE_DIR} clean
}

build_html () {
    sphinx-apidoc -f -o ${DOCS_SRC} ../nqft/
    make --directory=${MAKE_DIR} html
}

display_doc () {
    open ${HTML_SRC}/${PROJECT}.html
}

main () {
    clean_html
    build_html
    display_doc
}

main
