#!/bin/bash
# Installation as shown here: https://huggingface.co/spaces/ameerazam08/FLUX.1-dev-Inpainting-Model-Beta-GPU?docker=true
cd inpainting_server/flux_inpainter
python -m venv env
source env/bin/activate

if [ "$1" == "--install" ]; then
    python -m pip install -r requirements.txt
elif [ "$1" == "--run" ]; then
    python app.py
else
    echo "Usage: $0 [--install | --run]"
    exit 1
fi