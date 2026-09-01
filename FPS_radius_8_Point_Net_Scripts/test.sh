#!/bin/bash
#SBATCH --job-name=facebase_test
#SBATCH --output=/home/changj9/projects/facebase/FPS_5n5k_5n20/logs/test_r8_FPS_new_%j.out
#SBATCH --error=/home/changj9/projects/facebase/FPS_5n5k_5n20/logs/test_r8_FPS_new_%j.err


mkdir -p /home/changj9/projects/facebase/logs

#python /home/changj9/projects/facebase/FP/scripts/test.py
python /home/changj9/projects/facebase/FPS_5n5k_5n20/FPS_radius_8_Point_Net_Scripts/test_fps.py


#this was edited from r8 to call the test.py for r6
