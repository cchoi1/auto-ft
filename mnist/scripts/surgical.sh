#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=8G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="surgical" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate
cd ../

#echo "VAL ID"
#echo "PATIENCE 1"
#python3 main.py --method surgical --ft_id_dist brightness --ft_ood_dist impulse_noise --test_dist mnistc --optimizer_name LOptNet --num_nets 1 \
#--meta_steps 160 --patience 1 --val id --num_seeds 3
#echo "PATIENCE 3"
#python3 main.py --method surgical --ft_id_dist brightness --ft_ood_dist impulse_noise --test_dist mnistc --optimizer_name LOptNet --num_nets 1 \
#--meta_steps 160 --patience 3 --val id --num_seeds 3

echo "VAL OOD"
#echo "PATIENCE 1"
#python3 main.py --method surgical --ft_id_dist brightness --ft_ood_dist impulse_noise --test_dist mnistc --optimizer_name LOptNet --num_nets 1 \
#--meta_steps 160 --patience 1 --val ood --num_seeds 3
echo "PATIENCE 3"
python3 main.py --method surgical --ft_id_dist brightness --ft_ood_dist impulse_noise --test_dist mnistc --optimizer_name LOptNet --num_nets 1 \
--meta_steps 160 --patience 3 --val ood --num_seeds 3