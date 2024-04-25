#!/bin/bash

cd ProPainter

# Define the input directory containing subdirectories
input_dir="./../data/images_for_ppainter"

# Loop through each subdirectory in the input directory
for subdir in "$input_dir"/*; do
    if [ -d "$subdir" ]; then
        # Extract the base name of the subdirectory
        subdir_name=$(basename "$subdir")

        # Run inference_propainter.py script with appropriate arguments
        python inference_propainter.py \
            -i "$subdir/$subdir_name" \
            -m "$subdir/${subdir_name}_mask" \
            -o "$subdir"
    fi
done

cd ..

#chmod +x ./scripts/ppainter_inference.sh
#./scripts/ppainter_inference.sh
