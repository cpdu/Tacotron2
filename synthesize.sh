#!/bin/bash

outdir=outdir
ckpt_iter=8000
test_file=filelists/ljspeech/test.txt
hparams=

python3 -u synthesize.py -o $outdir/syn_$ckpt_iter -c $outdir/checkpoint_$ckpt_iter -t $test_file --hparams=distributed_run=False,$hparams 
