#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64GB # Request 16GB of memory
#SBATCH --gres=gpu:2 # Request one GPU
#SBATCH --job-name="imagenet-autoft-100inner-500ep-15000ex" # Name the job (for easier monitoring)
#SBATCH --output=imagenet-autoft-100inner-500ep-15000ex.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/yoonho/data \
--id ImageNet --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet,ImageNetC \
--num_ood_hp_examples 15000 --use_class_balanced_ood --inner_steps 100 --autoft_epochs 500 --ft_epochs 10 \
--lr 1e-5 --wd 0.1 --batch-size 128 --warmup_length 1000 --accumulation_steps 4 \
--load ./zeroshot/clip_vitb16_imagenet2.pt \
--ft_data /iris/u/cchoi1/Data/csv/imagenet.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero --template openai_imagenet_template