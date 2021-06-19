INPUT_PATH="<ABSOLUTE_PATH>/pretrained/initial_pwc.pth.tar"
OUTPUT_PATH=""
LOGS_PATH=""

PYTHON_CMD="python main.py 
    /cluster/project/infk/hilliges/lectures/mp21/project6/dataset 
    --dataset humanflow
    --name $OUTPUT_PATH
    --pretrained $INPUT_PATH
    --epochs 40
    --batch-size 8"

bsub -n 1 -W 48:00 -o $LOGS_PATH -R "rusage[mem=4096, ngpus_excl_p=1]" $PYTHON_CMD
watch -n 1 bjobs