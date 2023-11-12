#!/bin/bash
#SBATCH --job-name="iwildcam_data_transfer"
#SBATCH --output="iwildcam_data_transfer.log"
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

gsutil -m rsync -r /iris/u/yoonho/data/wilds/iwildcam_v2.0 gs://robust-ft2/iwildcam_v2.0
gsutil -m rsync -r /iris/u/yoonho/data/wilds/iwildcam_unlabeled_v1.0 gs://robust-ft2/iwildcam_unlabeled_v1.0

echo "Data transfer to GCS completed."