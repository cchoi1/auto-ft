#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=16G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="cifar-autoft-vitb16-10inner-100ep-2" # Name the job (for easier monitoring)
#SBATCH --output=cifar-autoft-vitb16-10inner-100ep-2.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

#python3 src/main.py --method autoft --model ViT-L/14 --data-location /iris/u/cchoi1/Data \
#--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
#--num_ood_hp_examples 190 --ft_epochs 10 \
#--autoft_epochs 100 --inner_steps 1 --lr 3e-5 --wd 0.1 --batch-size 64 --warmup_length 4000 --workers 4 \
#--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitl14_openai_cifar10.pt

python3 src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/cchoi1/Data \
--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10 \
--num_ood_hp_examples 100 --autoft_epochs 100 --inner_steps 10 --ft_epochs 20 \
--lr 1e-5 --wd 0.2 --batch-size 256 --accumulation_steps 1 --warmup_length 500 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_cifar10.pt \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero --relative_to_flyp \
--template simple_template \
--ft_data /iris/u/cchoi1/Data/csv/cifar10.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--workers 4