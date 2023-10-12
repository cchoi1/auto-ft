#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="iwildcam-autoft-layer-100inner-200ep" # Name the job (for easier monitoring)
#SBATCH --output=iwildcam-autoft-layer-100inner-200ep.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/main.py --method autoft --model ViT-B/16 --loss_type LayerwiseLoss --data-location /iris/u/cchoi1/Data \
--id IWildCamTrain --ood IWildCamOODVal --eval-datasets IWildCamIDVal,IWildCamIDTest,IWildCamOODTest \
--num_ood_hp_examples -1 --ft_epochs 20 \
--autoft_epochs 200 --inner_steps 100 --lr 1e-5 --wd 0.1 \
--batch-size 256 --warmup_length 500 \
--losses ce hinge entropy dcm flyp l1zero l2zero l1init l2init \
--layerwise_loss --layerwise_opt \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_iwildcam.pt