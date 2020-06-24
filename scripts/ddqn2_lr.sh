#!/usr/bin/env bash

source ../venv/bin/activate

cd ..

python train.py ddqn2 --batch-norm --lr 0.00001 &
