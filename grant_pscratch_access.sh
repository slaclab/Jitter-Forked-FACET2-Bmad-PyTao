#!/bin/bash

parent_dir="/pscratch/sd/m/maxvarv"
#sub_dirs=("Linac_phase_amp_jitter_2025_03_17" "L0B_phase_jitter" "L0B_amp_jitter" "L1_phase_jitter" "L1_amp_jitter" "L2_phase_jitter" "L2_amp_jitter" "L3_phase_jitter" "L3_amp_jitter")
sub_dirs=("twoBunch_linac_phase_amp_jitter")

users=("cropp" "tdali92" "aedelen" "majernik")

for user in ${users[@]}; do
	echo "$user"
	setfacl -m u:"$user":rx "$parent_dir"
	for str in ${sub_dirs[@]}; do
		echo "$str"
		setfacl -m u:"$user":rx -R "$parent_dir/$str"
	done
done
