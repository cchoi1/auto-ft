#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=16G # Request 16GB of memory
#SBATCH --gres=gpu:4 # Request one GPU
#SBATCH --job-name="wise-ft-time-run" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/wise-ft/"

cd ..

python src/wise_ft.py   \
    --train-dataset=ImageNet  \
    --epochs=10  \
    --lr=0.00003  \
    --batch-size=512  \
    --cache-dir=cache  \
    --model=ViT-B/32  \
    --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch  \
    --template=openai_imagenet_template  \
    --results-db=results.json  \
    --save=models/wiseft/ViTB32  \
    --data-location=/iris/u/yoonho/data \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0