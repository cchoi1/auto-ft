# 4-shot PatchCamelyon
python src/main.py \
--method autoft \
--model ViT-B/16 \
--load PATH_TO_ZEROSHOT_MODEL \
--template patchcamelyon_template \
--data-location DATA_DIR \
--id PatchCamelyonTrain \
--ood PatchCamelyonValHOpt --k 4 \
--eval-datasets PatchCamelyonValEarlyStopping,PatchCamelyonTest \
--ft_data DATA_DIR/patchcamelyon/train.csv --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--num_ood_hp_examples 8 \
--autoft_epochs 50 --inner_steps 10 \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero \
--ft_epochs 20 --lr 1e-5 --wd 0.1 --warmup_length 0 --accumulation_steps 1 \
--repeats 50 --autoft_repeats 5


# 16-shot PatchCamelyon
python src/main.py \
--method autoft \
--model ViT-B/16 \
--load PATH_TO_ZEROSHOT_MODEL \
--template patchcamelyon_template \
--data-location DATA_DIR \
--id PatchCamelyonTrain \
--ood PatchCamelyonValHOpt --k 16 \
--eval-datasets PatchCamelyonValEarlyStopping,PatchCamelyonTest \
--ft_data DATA_DIR/patchcamelyon/train.csv --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--num_ood_hp_examples 32 \
--autoft_epochs 50 --inner_steps 10 \
--losses ce hinge entropy dcm flyp l1zero l2zero l1init l2init \
--ft_epochs 20 --lr 1e-5 --wd 0.1 --warmup_length 0 --accumulation_steps 1 \
--repeats 50 --autoft_repeats 5


# 32-shot PatchCamelyon
python src/main.py \
--method autoft \
--model ViT-B/16 \
--load PATH_TO_ZEROSHOT_MODEL \
--template patchcamelyon_template \
--data-location DATA_DIR \
--id PatchCamelyonTrain \
--ood PatchCamelyonValHOpt --k 32 \
--eval-datasets PatchCamelyonValEarlyStopping,PatchCamelyonTest \
--ft_data DATA_DIR/patchcamelyon/train.csv --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--num_ood_hp_examples 64 \
--autoft_epochs 50 --inner_steps 10 \
--losses ce hinge entropy dcm flyp l1zero l2zero l1init l2init \
--ft_epochs 20 --lr 1e-5 --wd 0.1 --warmup_length 0 --accumulation_steps 1 \
--repeats 50 --autoft_repeats 5
