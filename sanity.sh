#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python scripts/train.py --num-worlds 8192 --num-updates 10000 --gpu-sim --ckpt-dir ckpts/sanity --reward-mode Dense1 --run-name test --num-bptt-chunks 1 --gamma 0.998 --profile-report