#!/bin/bash
#SBATCH --job-name="delete-CLOC"
#SBATCH --output="delete-CLOC.log"
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --account=iris
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send ma

cd /iris/u/cchoi1/Data
rm -rf CLOC
cd /iris/u/cchoi1
rm -rf continuallearning
