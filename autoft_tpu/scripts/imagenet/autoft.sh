#!/bin/bash
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft_tpu/"

python3 src/finetune3.py --method autoft --model ViT-B/16 \
--data-location /home/carolinechoi/data \
--id ImageNet --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet \
--ft_data /home/carolinechoi/data/ImageNet/imagenet.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--num_ood_hp_examples 15000 --use_class_balanced_ood \
--ft_epochs 10 --inner_steps 10 --autoft_epochs 1000 \
--lr 1e-5 --wd 0.1 --batch-size 16 --warmup_length 500 --accumulation_steps 4 \
--load /home/carolinechoi/data/zeroshot/clip_vitb16_imagenet.pt \
--load_hparams /home/carolinechoi/data/hparams/imagenet/is=100_evals=500_ex=15000_relce.json \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero