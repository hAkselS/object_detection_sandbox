import os

from google.cloud import storage 

import subprocess

# Pulls in annotations
command1 = [
    "gsutil", "-m", "rsync", "-r",
    "gs://nmfs_odp_pifsc/PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/large_2016_dataset/annotations_v2",
    "fish_no_fish/test_scripts/data_dump/annotations"
]

# Pulls images 
command2 = [
    "gsutil", "-m", "rsync", "-r",
    "gs://nmfs_odp_pifsc/PIFSC/ESD/ARP/pifsc-ai-data-repository/fish-detection/MOUSS_fish_detection_v1/datasets/large_2016_dataset/images",
    "fish_no_fish/test_scripts/data_dump/fish_dataset/val/images"
]


# Execute the command(s)
# subprocess.run(command1, check=True) # Use images exported in yolo format instead 
subprocess.run(command2, check=True)