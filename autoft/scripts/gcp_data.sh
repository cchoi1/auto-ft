#!/bin/bash
#SBATCH --job-name="imagenet-data-transfer"
#SBATCH --output="imagenet-data-transfer.log"
#SBATCH --exclude=iris1,iris2,iris3,iris4
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --account=iris
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send ma

echo "Running on node: $(hostname)"

cd /scr
mkdir cchoi1
cd cchoi1
mkdir ImageNet
rsync -av --include='*/' --exclude='*' /iris/u/yoonho/data/ImageNet/ /scr/cchoi1/ImageNet/
cd /iris/u/yoonho/data/ImageNet && \
find . -type f -print0 | parallel -j 8 -0 rsync -avzR {} /scr/cchoi1/ImageNet/

