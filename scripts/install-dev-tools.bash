#!/bin/bash
git config --global diff.submodule log
git config --global status.submodulesummary 1
git config --global submodule.recurse true
git submodule update --init --recursive

pip install -e serums