#!/usr/bin/env bash

source ../venv/bin/activate

cd ..

python train.py ddqn1 --curiosity --batch-norm &