#!/bin/bash
#SBATCH --job-name=facebase_voxel
#SBATCH --output=/home/changj9/projects/facebase/voxel_logs/prepare_voxel_%j.out
#SBATCH --error=/home/changj9/projects/facebase/voxel_logs/prepare_voxel_%j.err

mkdir -p /home/changj9/projects/facebase/voxel_logs

python /home/changj9/projects/facebase/scripts/prepare_voxel.py
