#!/bin/bash

for seed in 11 12 13 14 15
do
    for scene in LavaPath LavaButton SingleBlockButton ObstructedBlockButton BlockStack PatternMatch ChickenCoop Chase SingleButton
    do
        # Dense PPO
        #CUDA_VISIBLE_DEVICES=5 python scripts/go_ppo_world_copy.py --num-worlds 8192 --fp16 --gpu-sim --reward-mode Dense1 --run-name hello --seed $seed --num-steps 10000 --num-bins 1 --num-checkpoints 100 --seeds-per-checkpoint 512 --binning random --ckpt-dir ckpts/ppo_dense_$scene_$seed/ --num-updates 10000 --new-frac 0.001 --bin-reward-type none --bin-reward-boost 0.01 --num-bptt-chunks 1 --entropy-loss-coef 0.001 --lr 0.0002 --level-type $scene &
        # Sparse PPO
        #CUDA_VISIBLE_DEVICES=6 python scripts/go_ppo_world_copy.py --num-worlds 8192 --fp16 --gpu-sim --reward-mode PerLevel --run-name hello --seed $seed --num-steps 10000 --num-bins 1 --num-checkpoints 100 --seeds-per-checkpoint 512 --binning random --ckpt-dir ckpts/ppo_dense_$scene_$seed/ --num-updates 10000 --new-frac 0.001 --bin-reward-type none --bin-reward-boost 0.01 --num-bptt-chunks 1 --entropy-loss-coef 0.001 --lr 0.0002 --level-type $scene &
        # Count-based PPO with hash
        #CUDA_VISIBLE_DEVICES=2 python scripts/go_ppo_world_copy.py --num-worlds 8192 --fp16 --gpu-sim --reward-mode PerLevel --run-name hello --seed $seed --num-steps 10000 --num-bins 2400 --num-checkpoints 100 --seeds-per-checkpoint 512 --binning hash --ckpt-dir ckpts/ppo_dense_$scene_$seed/ --num-updates 10000 --new-frac 0.001 --bin-reward-type count --bin-reward-boost 0.01 --num-bptt-chunks 1 --entropy-loss-coef 0.001 --lr 0.0002 --level-type $scene &
        # Proper count-based
        # GPT PPO if it exists...
        # RND
        #CUDA_VISIBLE_DEVICES=7 python scripts/go_ppo_world_copy.py --num-worlds 8192 --fp16 --gpu-sim --reward-mode PerLevel --run-name hello --seed $seed --num-steps 10000 --num-bins 1 --num-checkpoints 100 --seeds-per-checkpoint 512 --binning random --ckpt-dir ckpts/ppo_dense_$scene_$seed/ --num-updates 10000 --new-frac 0.001 --bin-reward-type none --bin-reward-boost 0.01 --num-bptt-chunks 1 --entropy-loss-coef 0.001 --lr 0.0002 --level-type $scene --use-intrinsic-loss
        # Count-based PPO with dumb counts
        #CUDA_VISIBLE_DEVICES=2 python scripts/go_ppo_world_copy.py --num-worlds 8192 --fp16 --gpu-sim --reward-mode PerLevel --run-name hello --seed $seed --num-steps 10000 --num-bins 2400 --num-checkpoints 100 --seeds-per-checkpoint 512 --binning hash --ckpt-dir ckpts/ppo_dense_$scene_$seed/ --num-updates 10000 --new-frac 0.001 --bin-reward-type count --bin-reward-boost 0.01 --num-bptt-chunks 1 --entropy-loss-coef 0.001 --lr 0.0002 --level-type $scene
    done
done

