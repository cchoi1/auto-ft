#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64GB # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --cpus-per-task=4 # Request 8 CPUs for this task
#SBATCH --job-name="imagenet32-autoft-finetune-50inner-100ep-1000ex-regen2" # Name the job (for easier monitoring)
#SBATCH --output=imagenet32-autoft-finetune-50inner-100ep-1000ex-regen2.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/yoonho/data \
--id ImageNet32 --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet32 \
--num_ood_hp_examples 1000 --inner_steps 50 --autoft_epochs 100 --ft_epochs 10 \
--lr 1e-5 --wd 0.1 --batch-size 128 --warmup_length 1000 --accumulation_steps 4 \
--load ./zeroshot/clip_vitb16_imagenet2.pt \
--ft_data /iris/u/cchoi1/Data/csv/imagenet32_onecap.csv --val_data /iris/u/cchoi1/Data/csv/imagenet32_val_onecap.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero --template openai_imagenet_template \
--relative_to_flyp --regenerate_head \
--load_hparams /iris/u/cchoi1/robust-optimizer/autoft/saved/ImageNet32/autoft/oodImageNetC_cdefhllll_relflyp/no1000_afep100_is50_ftep10_bs256_wd0.1_lr1e-05_run1_seed0_ViT-B/16/hparams.json
# #SBATCH --exclude=iris1,iris2,iris3,iris4