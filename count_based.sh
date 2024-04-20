#!/bin/bash

# Count-based intrinsic on sparse reward
#python scripts/go_ppo.py --num-worlds 8192 --fp16 --gpu-sim --reward-mode PerLevel --run-name hello --seed 0 --num-steps 10000 --num-bins 10000 --num-checkpoints 100 --seeds-per-checkpoint 512 --binning y_pos_door --ckpt-dir ckpts/ppo_sparse_count/ --num-updates 10000 --new-frac 0.001 --bin-reward-type count --bin-reward-boost 0.01 --num-bptt-chunks 1 --entropy-loss-coef 0.001 --lr 0.0002 

#for seed in 0 1 2 3 4 5 6 7
#do
#    CUDA_VISIBLE_DEVICES=$seed python scripts/go_ppo_world_copy.py --num-worlds 8192 --fp16 --gpu-sim --reward-mode PerLevel --run-name hello --seed $seed --num-steps 10000 --num-bins 10000 --num-checkpoints 100 --seeds-per-checkpoint 512 --binning block_button_new --ckpt-dir ckpts/ppo_sparse_count/ --num-updates 10000 --new-frac 0.001 --bin-reward-type count --bin-reward-boost 0.01 --num-bptt-chunks 1 --entropy-loss-coef 0.001 --lr 0.0002 & 
    #--use-fixed-world &
#done

#for seed in 0 1 2 3 4 5 6 7
#do
#    CUDA_VISIBLE_DEVICES=$seed python scripts/go_ppo_world_copy.py --num-worlds 8192 --fp16 --gpu-sim --reward-mode PerLevel --run-name hello --seed $seed --num-steps 10000 --num-bins 1000 --num-checkpoints 100 --seeds-per-checkpoint 512 --binning y_pos_door_entities --ckpt-dir ckpts/ppo_sparse_count/ --num-updates 10000 --new-frac 0.001 --bin-reward-type count --bin-reward-boost 0.01 --num-bptt-chunks 1 --entropy-loss-coef 0.001 --lr 0.0002 & 
    #--use-fixed-world &
#done

#for seed in 0 1 2 3 4 5 6 7
#do
#    CUDA_VISIBLE_DEVICES=$seed python scripts/go_ppo_world_copy.py --num-worlds 8192 --fp16 --gpu-sim --reward-mode PerLevel --run-name hello --seed $seed --num-steps 10000 --num-bins 1200 --num-checkpoints 100 --seeds-per-checkpoint 512 --binning lava --ckpt-dir ckpts/ppo_sparse_count/ --num-updates 10000 --new-frac 0.001 --bin-reward-type count --bin-reward-boost 0.01 --num-bptt-chunks 1 --entropy-loss-coef 0.001 --lr 0.0002 & 
#done

#for seed in 0 1 2 3 4 5 6 7
#do
#    CUDA_VISIBLE_DEVICES=$seed python scripts/go_ppo_world_copy.py --num-worlds 8192 --fp16 --gpu-sim --reward-mode PerLevel --run-name hello --seed $seed --num-steps 10000 --num-bins 2400 --num-checkpoints 100 --seeds-per-checkpoint 512 --binning chicken --ckpt-dir ckpts/ppo_sparse_count/ --num-updates 10000 --new-frac 0.001 --bin-reward-type count --bin-reward-boost 0.01 --num-bptt-chunks 1 --entropy-loss-coef 0.001 --lr 0.0002 & 
#done

#for seed in 0 1 2 3 4 5 6 7
#do
#    CUDA_VISIBLE_DEVICES=$seed python scripts/go_ppo_world_copy.py --num-worlds 8192 --fp16 --gpu-sim --reward-mode PerLevel --run-name hello --seed $seed --num-steps 10000 --num-bins 2400 --num-checkpoints 100 --seeds-per-checkpoint 512 --binning pattern --ckpt-dir ckpts/ppo_sparse_count/ --num-updates 10000 --new-frac 0.001 --bin-reward-type count --bin-reward-boost 0.01 --num-bptt-chunks 1 --entropy-loss-coef 0.001 --lr 0.0002 & 
#done

for seed in 0 1 2 3 4 5 6 7
do
    CUDA_VISIBLE_DEVICES=$seed python scripts/go_ppo_world_copy.py --num-worlds 8192 --fp16 --gpu-sim --reward-mode PerLevel --run-name hello --seed $seed --num-steps 10000 --num-bins 2400 --num-checkpoints 100 --seeds-per-checkpoint 512 --binning lava_button --ckpt-dir ckpts/ppo_sparse_count_lava_button/ --num-updates 10000 --new-frac 0.001 --bin-reward-type count --bin-reward-boost 0.01 --num-bptt-chunks 1 --entropy-loss-coef 0.001 --lr 0.0002 & 
done
