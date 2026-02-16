#!/bin/bash

CONFIG_FILE="./configs/test_AST_UrbanSound8K.yaml"

CODEC="opus"
OUTPUT_DIR="./results/AST_UrbanSound8K_opus"
KBPS= "3.0 6.0" # 12.0 24.0"


echo "=========================================================="
echo "🚀 Starting Experiment: AST | UrbanSound8K | opus"
echo "=========================================================="

echo ">>> [1/3] Running Freq Mode..."
python forward_sal_main.py \
    --cfgs $CONFIG_FILE \
    --codec $CODEC \
    --output_dir $OUTPUT_DIR \
    --mode freq \
    --step 10 \
    --kbps_list $KBPS

#echo ">>> [2/3] Running Time Mode..."
#python forward_sal_main.py \
#    --cfgs $CONFIG_FILE \
#    --codec $CODEC \
#    --output_dir $OUTPUT_DIR \
#    --mode time \
#    --step 10 \
#    --kbps_list $KBPS

#echo ">>> [3/3] Running Pixel Mode..."
#python forward_sal_main.py \
#    --cfgs $CONFIG_FILE \
#    --codec $CODEC \
#    --output_dir $OUTPUT_DIR \
#    --mode pixel \
#    --step 5 \
#    --kbps_list $KBPS


echo "=========================================================="
echo "🎉 All modes (Freq, Time, Pixel) completed successfully!"
echo "=========================================================="