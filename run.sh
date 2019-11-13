#!/bin/bash

outdir=outdir
mkdir -p $outdir
n_gpu=1           # should be 1 or 2

sbatch -p gpu --mem=30G --ntasks-per-node 1 --gres=gpu:$n_gpu -o $outdir/log_%j.log train.sh $outdir $n_gpu
