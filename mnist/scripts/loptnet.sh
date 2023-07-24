#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=8G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="loptnet-mnist-loss-curves" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate
cd ../

# for param in 'p' 'g' 'p_norm' 'g_norm' 'g_norm_avg' 'depth' 'wb' 'dist_init_param' 'iter' 'iter_frac' 'loss' 'loss_ema' 'tensor_rank'
#for param in 'p' 'g' 'p_norm' 'g_norm' 'g_norm_avg' 'wb' 'dist_init_param' 'iter' 'iter_frac' 'loss' 'loss_ema' 'tensor_rank'
#do
#  echo "PARAM $param"
#  python3 main.py --method ours --pretrain_dist mnist --ft_id_dist brightness --ft_ood_dist impulse_noise --test_dist mnistc \
#  --optimizer_name LOptNet --num_nets 1 \
#  --inner_steps 5 --meta_steps 150 --val ood --no_wandb --features iter_frac $param
#done

#for INNERSTEPS in 5 10 20 30
#do
#  for METASTEPS in 80 100 150 200 250
#  do
#    echo "---------------------"
#    echo "INNER STEPS: $INNERSTEPS, METASTEPS: $METASTEPS"
#    echo "---------------------"
#    python3 main.py --method ours --pretrain_dist mnist --ft_id_dist brightness --ft_ood_dist impulse_noise --test_dist mnistc \
#    --optimizer_name LOptNet --num_nets 1 \
#    --inner_steps $INNERSTEPS --meta_steps $METASTEPS --val ood --no_wandb \
#    --features 'p' 'g' 'p_norm' 'g_norm' 'g_norm_avg' 'depth' 'dist_init_param' 'iter' 'loss' 'loss_ema' 'tensor_rank'
#  done
#done

#python3 main.py --method ours --pretrain_dist mnist --ft_id_dist brightness --ft_ood_dist impulse_noise --test_dist mnistc \
#--optimizer_name LOptNet --num_nets 1 \
#--inner_steps 5 --meta_steps 150 --patience 3 --val ood \
#--features 'iter_frac'

python3 main.py --method ours --pretrain_dist mnist --ft_id_dist brightness --ft_ood_dist impulse_noise --test_dist mnistc \
--optimizer_name LOptNet --num_nets 1 \
--inner_steps 5 --meta_steps 150 --patience 3 --val ood \
--features 'depth'