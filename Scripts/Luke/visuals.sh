NAME=fine_tune_27

python /cluster/home/lusmith/BananaPyjama/generate_visuals.py \
    /cluster/project/infk/hilliges/lectures/mp21/project6/dataset \
    --phase test \
    --pred-dir /cluster/home/lusmith/MachinePerception/submission/fine_tune_27 \
    --output-dir /cluster/home/lusmith/MachinePerception/visuals/fine_tune_27
# python /cluster/home/lusmith/BananaPyjama/generate_visuals.py /cluster/project/infk/hilliges/lectures/mp21/project6/dataset --phase test --output-dir /cluster/home/lusmith/MachinePerception/visuals/full_set