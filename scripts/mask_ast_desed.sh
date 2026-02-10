#!/bin/bash

DEVICE=0
CFG="configs/mask_AST_desed.yaml"
OUT="/data/dcase/dataset/audio_sal_forward_ast/"

# Opus — 3 parallel jobs
for kbps in 6.0 12.0 24.0; do
    CUDA_VISIBLE_DEVICES=$DEVICE python forward_sal_main.py \
        --cfgs $CFG --output_dir $OUT \
        --kbps_list $kbps --codec opus &
done
wait
echo "Opus done"

# # EnCodec — 5 parallel jobs
# for kbps in 1.5 3.0 6.0 12.0 24.0; do
#     CUDA_VISIBLE_DEVICES=$DEVICE python forward_sal_main.py \
#         --cfgs $CFG --output_dir $OUT \
#         --kbps_list $kbps --codec encodec &
# done
# wait
# echo "EnCodec done"