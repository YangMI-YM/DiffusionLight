if [ -z "$1" ]; then
  echo "Please provide a directory path. e.g. /home/yangmi/s3data-3/beauty-lvm/v2/light/768"
  exit 1
fi

# Store the directory path
directory="$1"
exposure_value='0,-0.5,-1.0,-1.5,-2.0,-2.5,-3.0,-3.5,-4.0,-4.5,-5'
output_directory="$2"
resolution=$(basename "$directory")
# Find all subfolders under the given directory
for subfolder in "$directory"/batch_*/; do
  # Check if it is a directory
  if [ -d "$subfolder" ]; then
    output_dir="$output_directory"/"$resolution"/$(basename "$subfolder")/brightness
    if [ "$(find "$output_dir" -mindepth 1 -print -quit)" ]; then
      # the folder is not empty, skip
      echo "Skipping $subfolder"
      continue
    fi
    # Pass the subfolder name to the Python command
    echo "Processing subfolder: $subfolder"
    
    #python ball2envmap.py --ball_dir "$subfolder"/square --envmap_dir "$subfolder"/envmap
    #python exposure2hdr.py --input_dir "$subfolder"/envmap --output_dir "$subfolder"/hdr --EV "$exposure_value" # optional
    #python square2chromeball.py --input_dir "$subfolder"/square --ball_dir "$subfolder"/ball
    #python exposure2hdr.py --input_dir "$subfolder"/ball --output_dir "$subfolder"/hdr_ball --EV "$exposure_value" # optional
    python bright_map.py --input_dir "$subfolder" --output_dir "$output_directory"
    #python light_probes_predict_v2.py --input_dir "$subfolder"/ball

    echo "Completed processing for: $subfolder"
  fi
done
