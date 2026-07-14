#!/usr/bin/env bash
#
####SBATCH --job-name=run_fmriprep
#SBATCH --output=fmriprep-%x-%j.out
#SBATCH --partition psych_gpu
#SBATCH --time 24:00:00
#SBATCH --mem 64GB
#SBATCH -n 1
#SBATCH -c 16


module load fmriprep/20.2.1

subjID=$1

. load_config.sh

bids_data_dir="${PATHS['BIDS_DATA_DIR']}"
output_dir="${PATHS['PREPROCESSED_DATA_DIR']}"
work_dir="${PATHS['FMRIPREP_WORK_DIR']}"


echo "bids_data_dir:" ${bids_data_dir}
make_dir $output_dir
make_dir $work_dir

start_time=$(date +%s)


fmriprep ${bids_data_dir} ${output_dir} participant --participant-label ${subjID} -w ${work_dir} \
--output-spaces MNI152Lin --bold2t1w-dof 6 --cifti-output 91k --n_cpus 16

end_time=$(date +%s)
duration=$((end_time - start_time))
formatted_duration=$(printf "%02d:%02d:%02d" $((duration / 3600)) $((duration % 3600 / 60)) $((duration % 60)))

echo "Script execution took $formatted_duration."
