#!/bin/bash
#SBATCH --job-name=facebase_train_mixed
#SBATCH --output=/home/changj9/projects/facebase/all_config_one_model/logs/train_mixed_%j.out
#SBATCH --error=/home/changj9/projects/facebase/all_config_one_model/logs/train_mixed_%j.err

mkdir -p /home/changj9/projects/facebase/all_config_one_model/logs

python /home/changj9/projects/facebase/all_config_one_model/scripts/train_mixed.py
