#!/bin/bash

#SBATCH --job-name=facebase_train
#SBATCH --output=/home/changj9/projects/facebase/FPS_5n5k_5n20/logs/train_r6_FPS_new_%j.out
#SBATCH --error=/home/changj9/projects/facebase/FPS_5n5k_5n20/logs/train_r6_FPS_new_%j.err

mkdir -p /home/changj9/projects/facebase/logs

python /home/changj9/projects/facebase/FPS_5n5k_5n20/FPS_radius_6_Point_Net_Scripts/train_fps.py
