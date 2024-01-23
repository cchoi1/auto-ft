#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64GB # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --cpus-per-task=4 # Request 8 CPUs for this task
#SBATCH --job-name="imagenet4-autoft-plot" # Name the job (for easier monitoring)
#SBATCH --output=imagenet4-autoft-plot.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/wise_ft.py --method autoft --model ViT-B/16 --data-location /iris/u/yoonho/data \
--id ImageNet4 --ood ImageNetV2 --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet \
--num_ood_hp_examples 1000 --inner_steps 5 --autoft_epochs 100 --ft_epochs 10 \
--lr 1e-5 --wd 0.1 --batch-size 128 --warmup_length 0 --accumulation_steps 4 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_imagenet2.pt,/iris/u/cchoi1/robust-optimizer/autoft/saved/ImageNet4/autoft/oodImageNetC_cdefhllll_relflyp/no1000_afep100_is5_ftep10_bs128_wd0.1_lr1e-05_run1_seed0_ViT-B/16/checkpoint_9.pt \
--ft_data /iris/u/cchoi1/Data/csv/imagenet4_onecap.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero --template openai_imagenet_template \
--relative_to_flyp --regenerate_head --workers 2 \
--alpha 1.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.95