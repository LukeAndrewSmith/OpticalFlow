PYTHON_CMD="python main.py 
    /cluster/project/infk/hilliges/lectures/mp21/project6/dataset 
    --dataset humanflow
    --name /cluster/home/bfreydt/MachinePerception/training/fine_tune_27
    --pretrained /cluster/home/bfreydt/BananaPyjama/pretrained/pwc_MHOF.pth.tar
    --epochs 27
    --batch-size 10"

bsub -I -n 1 -W 1:00 -R "rusage[mem=4096, ngpus_excl_p=1]" $PYTHON_CMD