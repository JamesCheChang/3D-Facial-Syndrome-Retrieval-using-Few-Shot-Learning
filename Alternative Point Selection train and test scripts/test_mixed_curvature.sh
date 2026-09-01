#!/bin/bash
#SBATCH --job-name=facebase_test_mixed
#SBATCH --output=/home/changj9/projects/facebase/all_config_one_model/logs/test_mixed_curvature_%j.out
#SBATCH --error=/home/changj9/projects/facebase/all_config_one_model/logs/test_mixed_curvature_%j.err

mkdir -p /home/changj9/projects/facebase/all_config_one_model/logs/

python /home/changj9/projects/facebase/all_config_one_model/scripts/test_mixed_curvature.py
