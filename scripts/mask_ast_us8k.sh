#!/bin/bash

DEVICE=1
CFG="configs/mask_AST_US8K.yaml"
OUT="/data/UrbanSound8k/audio_sal_forward_ast"
FOLDS="1"

# # Opus — 3 parallel jobs
for kbps in 6.0 12.0 24.0; do
    CUDA_VISIBLE_DEVICES=$DEVICE python forward_sal_main.py \
        --cfgs $CFG --output_dir $OUT --folds $FOLDS \
        --kbps_list $kbps --codec opus &
done
wait
echo "Opus done"

# EnCodec — 5 parallel jobs
for kbps in 1.5 3.0 6.0 12.0 24.0; do
    CUDA_VISIBLE_DEVICES=$DEVICE python forward_sal_main.py \
        --cfgs $CFG --output_dir $OUT --folds $FOLDS \
        --kbps_list $kbps --codec encodec &
done
wait
echo "EnCodec done"