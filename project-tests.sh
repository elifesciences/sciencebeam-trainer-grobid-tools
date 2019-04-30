#!/bin/sh

set -e

echo "running flake8"
python -m flake8 grobid_training tests

echo "running pylint"
python -m pylint grobid_training tests

echo "running pytest"
python -m pytest

echo "done"
