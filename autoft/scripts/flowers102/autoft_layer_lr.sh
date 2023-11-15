#!/bin/bash
#SBATCH --partition=iris # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:2 # Request one GPU
#SBATCH --job-name="flowers102-autoft-350inner-2000ep-400ex-class-balanced-layer-lr-xentobj" # Name the job (for easier monitoring)
#SBATCH --output=flowers102-autoft-350inner-2000ep-400ex-class-balanced-layer-lr-xentobj.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/cchoi1/Data \
--id Flowers102Train --ood Flowers102ValHOpt --eval-datasets Flowers102ValEarlyStopping,Flowers102Test \
--num_ood_hp_examples 400 --use_class_balanced_ood --ft_epochs 100 \
--autoft_epochs 2000 --inner_steps 350 --lr 1e-4 --wd 0.4 \
--batch-size 128 --accumulation_steps 2 --warmup_length 500 \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero \
--ft_data /iris/u/cchoi1/Data/flowers102/train.csv --template flowers102_template \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_flowers102.pt \
--relative_to_flyp --regenerate_head --xent_meta_objective \
--layerwise_opt
