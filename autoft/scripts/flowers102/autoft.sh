#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4
#SBATCH --time=72:00:00 # Reduced job length to 3 days, assuming optimization
#SBATCH --nodes=1 # Use one node (machine)
#SBATCH --mem=64G # 64GB of memory is typically sufficient, but adjust based on the model and batch size
#SBATCH --gres=gpu:4 # Request 4 GPUs for parallel processing
#SBATCH --cpus-per-task=16 # Increase the number of CPU cores per task to better handle data loading and preprocessing
#SBATCH --job-name="imagenet-autoft" # Renamed for clarity
#SBATCH --output=imagenet-autoft.log  # Renamed output log file
#SBATCH --mail-type=END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/cchoi1/Data \
--id Flowers102Train --ood Flowers102ValHOpt --eval-datasets Flowers102ValEarlyStopping,Flowers102Test \
--num_ood_hp_examples 600 --use_class_balanced_ood --ft_epochs 100 \
--autoft_epochs 500 --inner_steps 400 --lr 1e-4 --wd 0.4 \
--batch-size 128 --accumulation_steps 2 --warmup_length 500 \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero \
--ft_data /iris/u/cchoi1/Data/flowers102/train.csv --template flowers102_template \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_flowers102.pt \
--relative_to_flyp --regenerate_head \
--workers 2

# --load_hparams ./saved/Flowers102Train/autoft/oodFlowers102ValHOpt_cdefhllll_relflyp/no500_nouNone_afep500_is350_ftep100_bs128_wd0.2_lr1e-05_run1_seed0_ViT-B/16/hparams.json
