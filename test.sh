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

PYTHON_CMD="python test_humanflow.py 
    /cluster/project/infk/hilliges/lectures/mp21/project6/dataset
    --dataset humanflow 
    --div-flow 20 
    --no-norm 
    --pretrained /cluster/scratch/cguerner/mp/BananaPyjama/pretrained/pwc_MHOF.pth.tar  
    --output-dir /cluster/scratch/cguerner/mp/BananaPyjama/output/results-sample-baseline-leonhard"

echo "Running the following command $HOW:"
echo -e "\n$PYTHON_CMD\n"

if [ $USE_BSUB == 1 ]; then                 
        bsub -n 1 -W 1:00 -oo logs/log_test.txt -R "rusage[mem=4096, ngpus_excl_p=1]" $PYTHON_CMD
else        
        $PYTHON_CMD
fi
