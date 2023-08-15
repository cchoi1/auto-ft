#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=8G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="loptnet-colored_mnist" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate
cd ../..

python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
--ft_dists id --val ood --optimizer_name LOptNet --features g depth --lopt_net_dim 4 --output_channels 3 --seeds 0 \
--meta_batch_size 60 --inner_steps 10 --inner_steps_range 20 --meta_steps 300 --momentum --wnb

#python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
#--ft_dists id --val ood --optimizer_name LOptNet --features depth --lopt_net_dim 4 --output_channels 3 --seeds 0 \
#--meta_batch_size 60 --inner_steps 10 --inner_steps_range 20 --meta_steps 300 --meta_loss_avg_w 1.0 --meta_loss_final_w 0.0

#python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
#--ft_dists id --val ood --optimizer_name LOptNet --features depth --lopt_net_dim 16 --output_channels 3 --seeds 0 \
#--meta_batch_size 60 --inner_steps 10 --inner_steps_range 20 --meta_steps 300

#python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
#--ft_dists id --val ood --optimizer_name LOptNet --features depth --lopt_net_dim 4 --output_channels 3 --seeds 0 \
#--meta_batch_size 60 --inner_steps 10 --meta_steps 300 --wnb

#python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
#--ft_dists id --val ood --optimizer_name LOptNet --features depth --lopt_net_dim 4 --output_channels 3 --seeds 0 \
#--meta_batch_size 60 --inner_steps 30 --meta_steps 300 --wnb

#python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
#--ft_dists id --val ood --optimizer_name LOptNet --features depth --lopt_net_dim 4 --output_channels 3 --seeds 0 \
#--meta_batch_size 60 --inner_steps 10 --inner_steps_range 20 --meta_steps 300 --wnb

#python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
#--ft_dists id --val ood --optimizer_name LOptNet --features depth --lopt_net_dim 4 --output_channels 3 --seeds 0 \
#--meta_batch_size 100 --inner_steps 10 --inner_steps_range 20 --meta_steps 300 --wnb
#
#python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
#--ft_dists id --val ood --optimizer_name LOptNet --features depth --lopt_net_dim 4 --output_channels 3 --seeds 0 \
#--meta_batch_size 60 --inner_steps 20 --inner_steps_range 20 --meta_steps 300 --wnb
#
#python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
#--ft_dists id --val ood --optimizer_name LOptNet --features depth --lopt_net_dim 4 --output_channels 3 --seeds 0 \
#--meta_batch_size 60 --inner_steps 10 --inner_steps_range 40 --meta_steps 300 --wnb

#python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
#--ft_dists id --val ood --optimizer_name LOptNet --features depth --lopt_net_dim 4 --no_wandb --output_channels 3 --seeds 0 \
#--meta_batch_size 60 --inner_steps 10 --inner_steps_range 30 --meta_steps 300 --wnb --l2_lambda 1.0