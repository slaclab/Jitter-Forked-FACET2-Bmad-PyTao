#!/bin/bash

parent_dir="/pscratch/sd/m/maxvarv/Linac_phase_amp_jitter_2025_03_17"
target_dir="/global/cfs/cdirs/m4272/mvarverakis/2025-03-20_oneBunch_linac_phase_amp_jitter"
tarball_ext=".tar.gz"

mkdir -p "$target_dir"

for dir in "$parent_dir"/*; do
	# Check if the item is a directory
	if [ -d "$dir" ]; then

		# Extract the directory name
		dir_name=$(basename "$dir")
		
		if [ ! -f "$target_dir/$dir_name$tarball_ext" ]; then

			if [ -f  "$parent_dir/beams/activeBeamFile_$dir_name.h5" ]; then
				#echo "activeBeamFile_$dir_name.h5 in beams directory!"
				mv "$parent_dir/beams/activeBeamFile_$dir_name.h5" "$dir"
			fi

			if [ ! -f "$dir/$dir_name$tarball_ext" ]; then
				tar czf "$dir_name$tarball_ext" -C "$dir" .
				mv "$dir_name$tarball_ext" "$dir"
			fi		

			#echo "$dir/$dir_name$tarball_ext will be moved to $target_dir"
			cp "$dir/$dir_name$tarball_ext" "$target_dir"
	
		fi
	fi
done
