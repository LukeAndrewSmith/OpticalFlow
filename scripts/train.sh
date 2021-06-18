#!/bin/bash

DATA="/cluster/project/infk/hilliges/lectures/mp21/project6/dataset"
DATASET="humanflow"
PRETRAINED_MODEL_PATH="pretrained/model_best.pth.tar"
OUTPUT_PATH="output/testing_final_scripts_train"
LOG_FILE_PATH="logs/final_train_log.txt"
EPOCHS=3
BATCH_SIZE=8

PYTHON_CMD="python main.py 
    $DATA
    --dataset $DATASET
    --name $OUTPUT_PATH
    --pretrained $PRETRAINED_MODEL_PATH
    --epochs $EPOCHS
    --batch-size $BATCH_SIZE"

bsub -n 1 -W 48:00 -o $LOG_FILE_PATH -R "rusage[mem=4096, ngpus_excl_p=1]" $PYTHON_CMD
#watch -n 1 bjobs