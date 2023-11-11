#!/bin/bash
#SBATCH --job-name="gcs_data_transfer"
#SBATCH --output="gcs_data_transfer.log"
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --account=iris
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send ma

echo "Starting data transfer to Google Cloud Storage..."

# Transfer ImageNet-21k dataset
gsutil -m rsync -r /iris/u/yoonho/data/ImageNet/ILSVRC/Data/CLS-LOC/train gs://robust-ft2

echo "Data transfer to GCS completed."