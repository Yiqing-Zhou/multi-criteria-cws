#!/usr/bin/env bash

python ./make_dataset.py --training-data  final/all_train.txt --dev-data final/all_dev.txt \
--test-data final/all_test.txt -o dataset/final_all/dataset.pkl

#python ./make_dataset.py --training-data  data/$1/bmes/train-all.txt --dev-data data/$1/bmes/dev.txt \
#--test-data data/$1/bmes/test.txt -o dataset/$1/dataset.pkl