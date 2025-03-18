#!/bin/bash

ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ -z $1 ]; then
    exit
fi

mkdir -p ${ROOT_DIR}/ckpts
mkdir -p ${ROOT_DIR}/tb

rm -rf ${ROOT_DIR}/ckpts/$1
rm -rf ${ROOT_DIR}/tb/$1

XLA_PYTHON_CLIENT_PREALLOCATE=false MADRONA_MWGPU_KERNEL_CACHE=${ROOT_DIR}/build/cache python ${ROOT_DIR}/scripts/jax_train.py \
    --gpu-sim \
    --ckpt-dir ${ROOT_DIR}/ckpts \
    --tb-dir ${ROOT_DIR}/tb \
    --run-name $1 \
    --num-updates 5000 \
    --num-worlds 8192 \
    --lr 1e-3 \
    --steps-per-update 40 \
    --num-bptt-chunks 2 \
    --minibatch-size 1024 \
    --entropy-loss-coef 0.001 \
    --value-loss-coef 0.5 \
    --num-channels 512 \
    --pbt-ensemble-size 1 \
    --bf16 \
    --num-env-copies 1
    #--wandb \
