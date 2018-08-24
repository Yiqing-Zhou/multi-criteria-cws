#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 \
python main.py --dataset dataset/$1/dataset.pkl --num-epochs 10 \
--char-embeddings data/embedding/character.vec \
--output-dir output/$1 \
--dropout 0.5 \
--learning-rate 0.01 \
--learning-rate-decay 0.9 \
--hidden-dim 100 \
--bigram \
${@:2}