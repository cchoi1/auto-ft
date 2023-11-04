#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="sst2-autoft-32shot-5inner-20ep-5autoftreps" # Name the job (for easier monitoring)
#SBATCH --output=sst2-autoft-32shot-5inner-20ep-5autoftreps.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/cchoi1/Data \
--id sst2Train --ood sst2ValHOpt --k 32 --eval-datasets sst2ValEarlyStopping,sst2Test \
--num_ood_hp_examples 64 --ft_epochs 20 --autoft_epochs 20 --inner_steps 5 \
--lr 1e-5 --wd 0.2 --batch-size 256 --warmup_length 0 --accumulation_steps 1 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_sst22.pt \
--ft_data /iris/u/cchoi1/Data/csv/sst2/train.csv --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--losses ce dcm entropy hinge flyp l1init l1zero l2init l2zero --template sst2_template --repeats 50 --autoft_repeats 5
