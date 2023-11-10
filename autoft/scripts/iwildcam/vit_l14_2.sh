#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="iwildcam-autoft-vitl14-500inner-500ep-1000ex-class-balanced-relflyp-regen" # Name the job (for easier monitoring)
#SBATCH --output=iwildcam-autoft-vitl14-500inner-500ep-1000ex-class-balanced-relflyp-regen.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/main.py --method autoft --model ViT-L/14 --data-location /iris/u/cchoi1/Data \
--id IWildCamTrain --ood IWildCamOODVal --eval-datasets IWildCamIDVal,IWildCamIDTest,IWildCamOODTest \
--num_ood_hp_examples 1000 --use_class_balanced_ood --ft_epochs 20 --autoft_epochs 500 --inner_steps 500 \
--lr 1e-5 --wd 0.2 --batch-size 128 --warmup_length 500 --accumulation_steps 2 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_iwildcam2.pt --template iwildcam_template \
--ft_data /iris/u/cchoi1/Data/csv/iwildcam_v2.0/iwildcam.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero --relative_to_flyp --clip_gradient --regenerate_head