#!/bin/bash

for fold in {1..1}; do
    CUDA_VISIBLE_DEVICES=0 python beats_trainer.py \
        --configs configs/cls_AST_UrbanSound8K.yaml \
        --test_fold $fold
done