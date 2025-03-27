#!/bin/bash

# Check if a directory was provided as an argument
if [ -z "$1" ]; then
  echo "Please provide a directory path containing test images. e.g. ~/s3data/beauty-lvm/v2/cropped/1024"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Please provide an output directory. e.g. ~/volume/beauty-lvm/v2/light_mask"
  exit 1
fi

# Store the directory path
directory="$1"
output_directory="$2"
echo "Output directory confirmed: $output_directory."


# Pass the subfolder name to the Python command
echo "Processing folder: $directory"
CUDA_VISIBLE_DEVICES="$3" python inpaint.py --dataset "$directory" --output_dir "$output_directory" --ev='0,-2.5,-5'
python square2chromeball.py --input_dir "$output_directory"/square --ball_dir "$output_directory"/ball
python light_probes_predict_v2.py --input_dir "$output_directory"/ball
echo "Completed processing for: $directory"
