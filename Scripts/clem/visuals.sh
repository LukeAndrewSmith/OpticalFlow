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
PHASE="val"
PRED_DIR="/cluster/scratch/cguerner/mp/BananaPyjama/output/validation_test"
OUTPUT_DIR="/cluster/scratch/cguerner/mp/BananaPyjama/output/validation_test/visuals"

if [ -d "$OUTPUT_DIR" ]; then rm -rf $OUTPUT_DIR; fi

PYTHON_CMD="python generate_visuals.py 
    $DATA
    --phase $PHASE
    --pred-dir $PRED_DIR
    --output-dir $OUTPUT_DIR"

echo "Running the following command $HOW:"
echo -e "\n$PYTHON_CMD\n"

if [ $USE_BSUB == 1 ]; then                 
        bsub -n 1 -W 1:00 -oo logs/log_visuals.txt -R "rusage[mem=4096, ngpus_excl_p=1]" $PYTHON_CMD
        watch -n 1 bjobs
else        
        $PYTHON_CMD
fi