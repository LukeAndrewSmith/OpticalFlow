#!/bin/bash

DATA="/cluster/project/infk/hilliges/lectures/mp21/project6/dataset"
DATASET="humanflow"
PRETRAINED_MODEL_PATH=""
OUTPUT_DIR=""

if [ -d "$OUTPUT_DIR" ]; then rm -rf $OUTPUT_DIR; fi

PYTHON_CMD="python test_humanflow.py 
    $DATA
    --dataset $DATASET
    --div-flow 20 
    --no-norm 
    --pretrained $PRETRAINED_MODEL_PATH
    --output-dir $OUTPUT_DIR"

echo "Running the following command $HOW:"
echo -e "\n$PYTHON_CMD\n"

bsub -n 1 -W 1:00 -oo logs/log_test.txt -R "rusage[mem=4096, ngpus_excl_p=1]" $PYTHON_CMD
