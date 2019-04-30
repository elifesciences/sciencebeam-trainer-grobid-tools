#!/bin/sh

set -e

echo "running flake8"
python -m flake8 sciencebeam_trainer_grobid_tools tests

echo "running pylint"
python -m pylint sciencebeam_trainer_grobid_tools tests

echo "running pytest"
python -m pytest

echo "done"
