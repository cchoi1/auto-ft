#!/bin/bash
#SBATCH --job-name="imagenetc-data-transfer"
#SBATCH --output="imagenetc-data-transfer.log"
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --account=iris
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send ma

gsutil -m rsync -r /iris/u/cchoi1/Data/ImageNet-C gs://robust-ft2/ImageNet-C
