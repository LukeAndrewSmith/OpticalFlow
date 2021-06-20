#!/bin/bash

DATA="/cluster/project/infk/hilliges/lectures/mp21/project6/dataset"
DATASET="humanflow"
PRETRAINED_MODEL_PATH="pretrained/fine_tuned.pth.tar"
OUTPUT_DIR="output/testing_final_scripts_val"
LOG_FILE_PATH="logs/final_val_log.txt"

if [ -d "$OUTPUT_DIR" ]; then rm -rf $OUTPUT_DIR; fi

PYTHON_CMD="python val_humanflow.py 
    $DATA 
    --dataset $DATASET 
    --div-flow 20 
    --no-norm 
    --pretrained $PRETRAINED_MODEL_PATH 
    --output-dir $OUTPUT_DIR"

echo "Running the following command $HOW:"
echo -e "\n$PYTHON_CMD\n"
            
bsub -n 1 -W 1:00 -oo $LOG_FILE_PATH -R "rusage[mem=4096, ngpus_excl_p=1]" $PYTHON_CMD