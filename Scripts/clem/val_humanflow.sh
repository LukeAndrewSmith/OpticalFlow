#!/bin/bash

USE_BSUB=1

while [[ $# > 0 ]]
do
        case "$1" in
                -l|--local-run)
                        USE_BSUB=0
                        PARALLEL=""
                        HOW="locally"
                        ;;
        esac
        shift
done

DATA="/cluster/project/infk/hilliges/lectures/mp21/project6/dataset"
DATASET="humanflow"
PRETRAINED="/cluster/home/bfreydt/BananaPyjama/pretrained/model_best.pth.tar"
OUTPUT_DIR="/cluster/scratch/bfreydt/output/"

if [ -d "$OUTPUT_DIR" ]; then rm -rf $OUTPUT_DIR; fi

PYTHON_CMD="python val_humanflow.py 
    $DATA
    --dataset $DATASET 
    --div-flow 20 
    --no-norm 
    --pretrained $PRETRAINED
    --output-dir $OUTPUT_DIR"

echo "Running the following command $HOW:"
echo -e "\n$PYTHON_CMD\n"

if [ $USE_BSUB == 1 ]; then                 
        bsub -I -n 1 -W 1:00 -R "rusage[mem=4096, ngpus_excl_p=1]" $PYTHON_CMD
else        
        $PYTHON_CMD
fi
