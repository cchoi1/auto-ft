#!/bin/bash
#SBATCH --partition=iris # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:4 # Request one GPU
#SBATCH --job-name="iwildcam-id-autoft-50inner-500ep-1000ex-class-balanced-finetune" # Name the job (for easier monitoring)
#SBATCH --output=iwildcam-id-autoft-50inner-500ep-1000ex-class-balanced-finetune.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/cchoi1/Data \
--id IWildCamTrain --ood IWildCamIDVal --eval-datasets IWildCamIDVal,IWildCamIDTest,IWildCamOODTest \
--num_ood_hp_examples 1000 --use_class_balanced_ood --ft_epochs 20 --autoft_epochs 500 --inner_steps 50 \
--lr 1e-5 --wd 0.2 --batch-size 256 --warmup_length 500 --accumulation_steps 1 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_iwildcam2.pt --template iwildcam_template \
--ft_data /iris/u/cchoi1/Data/csv/iwildcam_v2.0/iwildcam.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero --clip_gradient --no_regenerate_head \
--load_hparams /iris/u/cchoi1/robust-optimizer/autoft/hparams/IWildCamID/50is_500os_1000ex_relce.json