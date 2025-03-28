#!/bin/bash

parent_dir="/pscratch/sd/m/maxvarv/Linac_phase_amp_jitter_2025_03_17"
target_dir="/global/cfs/cdirs/m4272/mvarverakis/2025-03-20_oneBunch_linac_phase_amp_jitter"

mkdir -p "$target_dir"

for dir in "$parent_dir"/*; do
	# Check if the item is a directory and not the 'beams' directory
	if [ -d "$dir" ] && [ "$(basename "$dir")" != "beams" ]; then
		# Check if the PENT.h5 file exists in the directory
		if [ -f "$dir/PENT.h5" ]; then
			# Extract the directory name
			dir_name=$(basename "$dir")
			# Set the target file name
			target_file="$target_dir/${dir_name}_PENT.h5"
			# Copy and rename the file
			cp "$dir/PENT.h5" "$target_file"
			#echo "Copied $dir/PENT.h5 to $target_file"
		fi
	fi
done
