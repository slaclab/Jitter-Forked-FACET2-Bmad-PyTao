#!/bin/bash

parent_dir="/pscratch/sd/m/maxvarv/"
target_dir="/global/cfs/cdirs/m4272/mvarverakis/"
sub_dirs=("L0B_phase_jitter" "L0B_amp_jitter" "L1_phase_jitter" "L1_amp_jitter" "L2_phase_jitter" "L2_amp_jitter" "L3_phase_jitter" "L3_amp_jitter")
tarball_ext=".tar.gz"

for subdir in ${sub_dirs[@]}; do
	echo "$subdir"

	full_target_dir="$target_dir$subdir"
	mkdir -p "$full_target_dir"
	
	cp "$parent_dir$subdir/jitter_reference.csv" "$full_target_dir"

	for dir in "$parent_dir$subdir"/*; do
		# Check if the item is a directory
		if [ -d "$dir" ] && [ "$(basename "$dir")" != "beams" ]; then

			# Extract the directory name
			dir_name=$(basename "$dir")
			
			if [ ! -f "$full_target_dir/$dir_name$tarball_ext" ]; then

				if [ -f  "$parent_dir$subdir/beams/activeBeamFile_$dir_name.h5" ]; then
					#echo "activeBeamFile_$dir_name.h5 in beams directory!"
					mv "$parent_dir$subdir/beams/activeBeamFile_$dir_name.h5" "$dir"
				fi

				if [ ! -f "$dir/$dir_name$tarball_ext" ]; then
					tar czf "$dir_name$tarball_ext" -C "$dir" .
					mv "$dir_name$tarball_ext" "$dir"
				fi		

				#echo "$dir/$dir_name$tarball_ext will be moved to $target_dir"
				cp "$dir/$dir_name$tarball_ext" "$full_target_dir"
			fi
		fi
	done
done
