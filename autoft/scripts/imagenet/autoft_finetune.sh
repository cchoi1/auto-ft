#!/bin/bash
#SBATCH --partition=iris # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Use one node (machine)
#SBATCH --mem=64G # 64GB of memory is typically sufficient, but adjust based on the model and batch size
#SBATCH --gres=gpu:4 # Request 4 GPUs for parallel processing
#SBATCH --cpus-per-task=4 # Increase the number of CPU cores per task to better handle data loading and preprocessing
#SBATCH --job-name="imagenet-autoft-finetune-regen-100is-100os-1000ex" # Renamed for clarity
#SBATCH --output=imagenet-autoft-finetune-regen-100is-100os-1000ex.log  # Renamed output log file
#SBATCH --mail-type=END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/yoonho/data \
--id ImageNet --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet \
--num_ood_hp_examples 1000 --inner_steps 100 --autoft_epochs 20 --ft_epochs 10 \
--lr 1e-5 --wd 0.1 --batch-size 512 --warmup_length 1000 --accumulation_steps 1 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_imagenet2.pt \
--ft_data /iris/u/cchoi1/Data/csv/imagenet_onecap.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero --template openai_imagenet_template \
--workers 4 --relative_to_flyp --regenerate_head --no_regenerate_head \
--load_hparams ./saved/ImageNet/autoft/oodImageNetC_cdefhllll/no1000_afep100_is100_ftep10_bs512_wd0.1_lr1e-05_run1_seed0_ViT-B/16/hparams.json
# --load_hparams /iris/u/cchoi1/robust-optimizer/autoft/hparams/ImageNet/100is_100os_15000ex_relflyp.json
# --load_hparams /iris/u/cchoi1/robust-optimizer/autoft/hparams/ImageNet/100is_20os_1000ex_relflyp.json