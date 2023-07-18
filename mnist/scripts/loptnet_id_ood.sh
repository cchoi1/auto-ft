#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=8G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="loptnet-id-ood" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate
cd ../

LIST=('p', 'g', 'depth', 'wb', 'dist_init_param', 'loss') # Define the input list

# Generate all possible subsets of the list
subsets=()
for ((i = 0; i < ${#LIST[@]}; i++)); do
    for ((j = i + 1; j <= ${#LIST[@]}; j++)); do
        subset=(${LIST[@]:i:j-i})
        subsets+=("${subset[*]}")
    done
done

# Execute the Python script with --features argument for each subset
for subset in "${subsets[@]}"; do
    echo $subset
    python3 main.py --method ours --ft_id_ood --ft_id_dist brightness --ft_ood_dist impulse_noise --test_dist mnistc \
    --optimizer_name LOptNet --num_nets 1 \
    --inner_steps 5 --meta_steps 150 --patience 3 --val ood --features "${subset}"
done