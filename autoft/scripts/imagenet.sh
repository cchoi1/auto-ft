python src/main.py \
--method autoft \
--model ViT-B/16 \
--load PATH_TO_ZEROSHOT_MODEL \
--template openai_imagenet_template \
--data-location DATA_DIR \
--id ImageNet \
--ood ImageNetC \
--eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet \
--ft_data DATA_DIR/csv/imagenet.csv --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--num_ood_hp_examples 15000 --use_class_balanced_ood \
--autoft_epochs 500 --inner_steps 100 \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero  \
--ft_epochs 10 --lr 1e-5 --wd 0.1 --batch-size 512 --warmup_length 0

python src/main.py \
--method autoft \
--model ViT-B/16 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_imagenet2.pt \
--template openai_imagenet_template \
--data-location /iris/u/yoonho/data \
--id ImageNet \
--ood ImageNetC \
--eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet \
--ft_data /iris/u/cchoi1/Data/csv/imagenet.csv --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--num_ood_hp_examples 15000 --use_class_balanced_ood \
--autoft_epochs 500 --inner_steps 100 \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero  \
--ft_epochs 10 --lr 1e-5 --wd 0.1 --batch-size 512 --warmup_length 0
