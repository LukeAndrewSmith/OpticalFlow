#!/bin/bash

#module load gcc/6.3.0 python_gpu/3.8.5 cuda/10.1.243 eth_proxy
#module load python_gpu/3.6.4 hdf5 eth_proxy
#module load cudnn/7.2

module load gcc/6.3.0 python_gpu/3.8.5

#rm -r venv

if [ ! -d "venv/" ]; then
    python3 -m venv --system-site-package venv
    echo "Created virtual environment."
fi

export TA_CACHE_DIR="/scratch/$USER/.cache"

source venv/bin/activate
python3 -m pip install --upgrade pip
pip3 install --default-timeout=100 -r requirements.txt
#pip3 install tensorboardX>=1.4
