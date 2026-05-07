#!/bin/bash

lrs=(0.0001 0.001 0.01)
wds=(0.0 0.01 0.1)

for lr in "${lrs[@]}"
do
  for wd in "${wds[@]}"
  do
    echo "Running lr=$lr weight_decay=$wd"

    python ggs/train_predictor.py \
      experiment=train/COV-oracle \
      optimizer.lr=$lr \
      optimizer.weight_decay=$wd

  done
done
