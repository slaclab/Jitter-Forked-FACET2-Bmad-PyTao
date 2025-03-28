#!/bin/bash

parent_dir="/pscratch/sd/m/maxvarv/L1_phase_jitter"

for dir in "$parent_dir"/*; do
	# Check if the item is a directory
	if [ -d "$dir" ]; then

		# Extract the directory name
		dir_name=$(basename "$dir")
		
		if [ -f  "$parent_dir/activeBeamFile_$dir_name.h5" ]; then
			#echo "$parent_dir/activeBeamFile_$dir_name.h5 found in parent dir!"
			mv "$parent_dir/activeBeamFile_$dir_name.h5" "$dir"
		fi
	fi
done
