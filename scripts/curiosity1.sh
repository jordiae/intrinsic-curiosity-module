#!/usr/bin/env bash

source ../venv/bin/activate

cd ..

python train.py curiosity --curiosity --batch-norm &
