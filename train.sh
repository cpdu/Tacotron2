#!/bin/bash

outdir=$1
n_gpu=$2

ckpt_iter=
hparams=

if [ $ngpu -eq 1  ]; then
  python3 -u train.py --n_gpus 1 -o $outdir --hparams=distributed_run=False,$hparams
  # python3 -u train.py -c $outdir/checkpoint_$ckpt_iter --n_gpus 1 -o $outdir --hparams=distributed_run=False,$hparams
elif [ $ngpu -eq 2 ]; then
  python3 -u train.py --n_gpus 2 --rank 0 -o $outdir --hparams=distributed_run=True,$hparams &
  python3 -u train.py --n_gpus 2 --rank 1 -o $outdir --hparams=distributed_run=True,$hparams
  # python3 -u train.py --n_gpus 2 --rank 0 -c $outdir/checkpoint_$ckpt_iter -o $outdir --hparams=distributed_run=True,$hparams &
  # python3 -u train.py --n_gpus 2 --rank 1 -c $outdir/checkpoint_$ckpt_iter -o $outdir --hparams=distributed_run=True,$hparams
fi
