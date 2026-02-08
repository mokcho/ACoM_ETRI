export WANDB_API_KEY=[YOUR_WANDB_KEY]

#!/bin/bash

for fold in {1..5}; do
    CUDA_VISIBLE_DEVICES=0 python beats_trainer.py \
        --configs configs/cls_AST_ESC-50.yaml \
        --test_fold $fold
done