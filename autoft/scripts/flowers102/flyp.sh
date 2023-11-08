python3 src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/cchoi1/Data \
--id Flowers102Train --ood Flowers102ValHOpt --eval-datasets Flowers102ValEarlyStopping,Flowers102Test \
--num_ood_hp_examples 1000 --use_class_balanced_ood --ft_epochs 100 \
--autoft_epochs 500 --inner_steps 100 --lr 1e-5 --wd 0.0 \
--batch-size 128 --accumulation_steps 2 --warmup_length 500 \
--losses flyp \
--ft_data /iris/u/cchoi1/Data/flowers102/train.csv --template flowers102_template \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_flowers102.pt \
--workers 2