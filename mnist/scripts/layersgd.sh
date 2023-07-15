#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=8G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="layer-sgd-hyperparam-tune" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate
cd ../

#for INNERSTEPS in 5 10 20 30
#do
#  for METASTEPS in 80 100 150 200 250
#  do
#    echo "---------------------"
#    echo "INNER STEPS: $INNERSTEPS, METASTEPS: $METASTEPS"
#    echo "---------------------"
#    python3 main.py --method ours --ft_id_dist brightness --ft_ood_dist impulse_noise --test_dist mnistc \
#    --optimizer_name LayerSGD --num_nets 1 \
#    --meta_steps $METASTEPS --inner_steps $INNERSTEPS --patience 3 --val ood --num_seeds 3
#  done
#done

python3 main.py --method ours --ft_id_dist brightness --ft_ood_dist impulse_noise --test_dist mnistc \
--optimizer_name LayerSGD --num_nets 1 \
--inner_steps 5 --meta_steps 150 --patience 3 --val ood --num_seeds 3