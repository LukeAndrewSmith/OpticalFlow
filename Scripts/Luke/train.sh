PYTHON_CMD="python main.py 
    /cluster/project/infk/hilliges/lectures/mp21/project6/dataset 
    --dataset humanflow
    --name /cluster/home/lusmith/MachinePerception/training/fine_tune_27
    --pretrained /cluster/home/lusmith/BananaPyjama/pretrained/pwc_MHOF.pth.tar
    --epochs 27
    --batch-size 10"

bsub -n 1 -W 24:00 -o /cluster/home/lusmith/MachinePerception/training/fine_tune_27/logs/log_train -R "rusage[mem=4096, ngpus_excl_p=1]" $PYTHON_CMD
watch -n 1 bjobs