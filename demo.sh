#!/bin/bash
export PYTHONPATH="$PWD:$PYTHONPATH"

python demo/demo_unidac${1:+_$1}.py --model-file checkpoints/unidac.pt