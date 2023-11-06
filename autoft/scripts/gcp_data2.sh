#!/bin/bash
#SBATCH --job-name="gcs_data_transfer2"
#SBATCH --output="gcs_data_transfer2.log"
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --account=iris
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send ma

echo "Starting data transfer to Google Cloud Storage..."

# Transfer objectnet-1.0 dataset
gsutil -m rsync -r /iris/u/cchoi1/Data/objectnet-1.0 gs://robust-ft2

# Transfer ImageNet-A dataset
gsutil -m rsync -r /iris/u/cchoi1/Data/ImageNet-A gs://robust-ft2

# Transfer ImageNet-R dataset
gsutil -m rsync -r /iris/u/cchoi1/Data/ImageNet-R gs://robust-ft2

# Transfer ImageNet-C dataset
gsutil -m rsync -r /iris/u/cchoi1/Data/ImageNet-C gs://robust-ft2

# Transfer ImageNet-V2 dataset
gsutil -m rsync -r /iris/u/cchoi1/Data/ImageNet-V2 gs://robust-ft2

# Transfer Sketch dataset
gsutil -m rsync -r /iris/u/cchoi1/Data/sketch gs://robust-ft2

# Transfer ImageNet-21k dataset
gsutil -m rsync -r /iris/u/yoonho/data/ImageNet gs://robust-ft2

echo "Data transfer to GCS completed."

