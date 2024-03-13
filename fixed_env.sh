#!/bin/bash

for seed in 0 1 2 3 4 5 6 7
do
    CUDA_VISIBLE_DEVICES=$seed python scripts/go_ppo.py --num-worlds 8192 --fp16 --reward-mode Dense2 --run-name hello --num-steps 10000 --num-bins 1000 --num-checkpoints 1000 --seeds-per-checkpoint 512 --binning block_button --ckpt-dir ckpts/block_button_seed_$seed --num-updates 10000 --bin-reward-type none --bin-reward-boost 0.01 --num-bptt-chunks 1 --new-frac 0.95 --sampling-strategy uniform --gpu-sim --steps-per-update 40 --make-graph --use-fixed-world --seed $seed &
done

python scripts/go_ppo.py --num-worlds 8192 --fp16 --reward-mode PerLevel --run-name hello --num-steps 10000 --num-bins 1000 --num-checkpoints 1000 --seeds-per-checkpoint 512 --binning block_button --ckpt-dir ckpts/block_button_seed_$seed --num-updates 10000 --bin-reward-type count --bin-reward-boost 0.01 --num-bptt-chunks 1 --new-frac 0.95 --sampling-strategy uniform --gpu-sim --steps-per-update 40 --make-graph --use-fixed-world --seed 0 &