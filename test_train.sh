if [ -z $1 ]
then
    echo "Specify a username as argument"
fi
bsub -n 1 -W 1:00 -o /cluster/home/$1/log_main -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py /cluster/project/infk/hilliges/lectures/mp21/project6/dataset/ --dataset "humanflow" --pretrained /cluster/home/$1/BananaPyjama/pretrained/pwc_MHOF.pth.tar
