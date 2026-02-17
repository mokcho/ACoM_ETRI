#!/bin/bash
set -e

ALPHAS="0.5 0.8 1.0 1.2 1.5"
FOLDS="1 2 3 4 5"
MODES="freq"
IMPORTANCE="saliency"

declare -A CODEC_KBPS=(
    ["opus"]="6.0 12.0 24.0"
    ["encodec"]="1.5 3.0 6.0 12.0 24.0"
)
declare -A CFGS=(
    ["AST"]="configs/mask_AST_ESC-50.yaml"
    ["BEATs"]="configs/mask_BEATs_ESC-50.yaml"
)
declare -A OUTDIRS=(
    ["AST"]="/data/ESC-50-master/audio_sal_forward_ast"
    ["BEATs"]="/data/ESC-50-master/audio_sal_forward_beats"
)

for MODEL in AST BEATs; do
    for CODEC in opus encodec; do
        echo "=== ${MODEL} / ${CODEC} ==="
        python forward_sal_th.py \
            --importance $IMPORTANCE \
            --alpha $ALPHAS \
            --pruning_modes $MODES \
            --cfgs "${CFGS[$MODEL]}" \
            --output_dir "${OUTDIRS[$MODEL]}" \
            --codec "$CODEC" \
            --kbps_list ${CODEC_KBPS[$CODEC]} \
            --folds $FOLDS
    done
done