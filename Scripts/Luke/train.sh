# if [ -z $1 ]
# then
#     echo "Specify a username as argument"
# fi
bsub -n 1 -W 1:00 -o /cluster/home/lusmith/MachinePerception/training/log_train -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py /cluster/project/infk/hilliges/lectures/mp21/project6/dataset --dataset "humanflow" --name '/cluster/home/lusmith/MachinePerception/training' --pretrained /cluster/home/lusmith/BananaPyjama/pretrained/pwc_MHOF.pth.tar
watch -n 1 bjobs