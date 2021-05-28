PYTHON_CMD="python main.py 
    /cluster/project/infk/hilliges/lectures/mp21/project6/dataset 
    --dataset humanflow
    --name /cluster/home/lusmith/MachinePerception/training/test_training/validate
    --pretrained /cluster/home/lusmith/MachinePerception/training/test_training/checkpoint.pth.tar
    -e"
    # --epoch_size 200
    # --batch-size 10

bsub -n 1 -W 10:00 -o /cluster/home/lusmith/MachinePerception/training/test_training/logs/log_validate -R "rusage[mem=4096, ngpus_excl_p=1]" $PYTHON_CMD
watch -n 1 bjobs