#!/bin/bash

module load gcc/6.3.0 python_gpu/3.8.5

if [ ! -d "venv/" ]; then
    python3 -m venv --system-site-package venv
    echo "Created virtual environment."
fi

export TA_CACHE_DIR="/scratch/$USER/.cache"

source venv/bin/activate
python3 -m pip install --upgrade pip
pip3 install --default-timeout=100 -r requirements.txt
