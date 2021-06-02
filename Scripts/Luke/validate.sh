PYTHON_CMD="python main.py 
    /cluster/project/infk/hilliges/lectures/mp21/project6/dataset 
    --dataset humanflow
    --name /cluster/home/lusmith/MachinePerception/training/fine_tune
    --pretrained /cluster/home/lusmith/MachinePerception/training/fine_tune/checkpoint.pth.tar
    -e"

bsub -n 1 -W 01:00 -o /cluster/home/lusmith/MachinePerception/training/fine_tune/logs/log_validate -R "rusage[mem=4096, ngpus_excl_p=1]" $PYTHON_CMD
watch -n 1 bjobs