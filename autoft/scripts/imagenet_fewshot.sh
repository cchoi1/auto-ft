# 4-shot ImageNet
python src/main.py \
--method autoft \
--model ViT-B/16 \
--load PATH_TO_ZEROSHOT_MODEL \
--template openai_imagenet_template \
--data-location DATA_DIR \
--id ImageNet4 \
--ood ImageNetV2 \
--eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet4 \
--ft_data DATA_DIR/csv/imagenet4.csv --val_data DATA_DIR/csv/imagenet4_val.csv --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--num_ood_hp_examples 1000 \
--hopt_evals 100 --inner_steps 5 \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero \
--ft_epochs 10 --lr 1e-5 --wd 0.1 --warmup_length 1000

# 16-shot ImageNet
python src/main.py \
--method autoft \
--model ViT-B/16 \
--load PATH_TO_ZEROSHOT_MODEL \
--template openai_imagenet_template \
--data-location DATA_DIR \
--id ImageNet16 \
--ood ImageNetC \
--eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet,ImageNetC \
--ft_data DATA_DIR/csv/imagenet16.csv --val_data DATA_DIR/csv/imagenet16_val.csv --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--num_ood_hp_examples 1000 \
--hopt_evals 100 --inner_steps 20 \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero --relative_to_flyp --regenerate_head \
--ft_epochs 10 --lr 1e-5 --wd 0.1 --warmup_length 1000

# 32-shot ImageNet
python src/main.py \
--method autoft \
--model ViT-B/16 \
--load PATH_TO_ZEROSHOT_MODEL \
--template openai_imagenet_template \
--data-location DATA_DIR \
--id ImageNet32 \
--ood ImageNetC \
--eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet32,ImageNetC \
--ft_data DATA_DIR/csv/imagenet32.csv --val_data DATA_DIR/csv/imagenet32_val.csv --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--num_ood_hp_examples 1000 \
--hopt_evals 100 --inner_steps 50 \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero --relative_to_flyp --regenerate_head \
--ft_epochs 10 --lr 1e-5 --wd 0.1 --warmup_length 1000
