#!/bin/bash
#SBATCH --job-name="del_folders"
#SBATCH --output="del_folders.log"
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --account=iris
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send ma

echo "Starting data deletion in Google Cloud Storage..."
gsutil -m rm -rf $(gsutil ls gs://robust-ft2/n*)
echo "Data deletion completed."