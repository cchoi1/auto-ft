#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64GB # Request 16GB of memory
#SBATCH --gres=gpu:2 # Request one GPU
#SBATCH --cpus-per-task=4 # Request 8 CPUs for this task
#SBATCH --job-name="cifar-vitb16-flyp-2" # Name the job (for easier monitoring)
#SBATCH --output=cifar-vitb16-flyp-2.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/yoonho/data \
--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10 \
--num_ood_hp_examples -1 --inner_steps 1 --autoft_epochs 1 --ft_epochs 20 \
--lr 1e-5 --wd 0.2 --batch-size 256 --warmup_length 500 --accumulation_steps 1 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_cifar10.pt \
--ft_data /iris/u/cchoi1/Data/csv/cifar10.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--losses flyp --template simple_template \
--relative_to_flyp --workers 4