#!/bin/bash
#SBATCH --job-name=facebase_curvature
#SBATCH --output=/home/changj9/projects/facebase/curvature_logs/prepare_curvature_%j.out
#SBATCH --error=/home/changj9/projects/facebase/curvature_logs/prepare_curvature_%j.err

mkdir -p /home/changj9/projects/facebase/curvature_logs

python /home/changj9/projects/facebase/scripts/prepare_curvature.py
