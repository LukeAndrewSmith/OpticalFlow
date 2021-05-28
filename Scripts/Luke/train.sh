PYTHON_CMD="python main.py 
    /cluster/project/infk/hilliges/lectures/mp21/project6/dataset 
    --dataset humanflow
    --name /cluster/home/lusmith/MachinePerception/training/test_training
    --pretrained /cluster/home/lusmith/BananaPyjama/pretrained/pwc_MHOF.pth.tar
    --epochs 5"
    # --epoch_size 200
    # --batch-size 10

bsub -n 1 -W 10:00 -o /cluster/home/lusmith/MachinePerception/training/multi_epoch_training/logs/log_train -R "rusage[mem=4096, ngpus_excl_p=1]" $PYTHON_CMD
watch -n 1 bjobs