# 4-shot SST2
python3 src/main.py \
--method autoft \
--model ViT-B/16 \
--load PATH_TO_ZEROSHOT_MODEL \
--template sst2_template \
--data-location DATA_DIR \
--id sst2Train \
--ood sst2ValHOpt --k 4 \
--eval-datasets sst2ValEarlyStopping,sst2Test \
--ft_data DATA_DIR/csv/sst2/train.csv --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--num_ood_hp_examples 8 \
--autoft_epochs 50 --inner_steps 10 \
--losses ce dcm entropy hinge flyp l1init l1zero l2init l2zero \
--ft_epochs 20 --lr 1e-5 --wd 0.2 --warmup_length 0 \
--repeats 50 --autoft_repeats 5


# 16-shot SST2
python3 src/main.py \
--method autoft \
--model ViT-B/16 \
--load PATH_TO_ZEROSHOT_MODEL \
--template sst2_template \
--data-location DATA_DIR \
--id sst2Train \
--ood sst2ValHOpt --k 16 \
--eval-datasets sst2ValEarlyStopping,sst2Test \
--ft_data DATA_DIR/csv/sst2/train.csv --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--num_ood_hp_examples 32 \
--autoft_epochs 50 --inner_steps 10 \
--losses ce dcm entropy hinge flyp l1init l1zero l2init l2zero \
--ft_epochs 20 --lr 1e-5 --wd 0.2 --warmup_length 0 \
--repeats 50 --autoft_repeats 5


# 32-shot SST2
python3 src/main.py \
--method autoft \
--model ViT-B/16 \
--load PATH_TO_ZEROSHOT_MODEL \
--template sst2_template \
--data-location DATA_DIR \
--id sst2Train \
--ood sst2ValHOpt --k 32 \
--eval-datasets sst2ValEarlyStopping,sst2Test \
--ft_data DATA_DIR/csv/sst2/train.csv --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--num_ood_hp_examples 64 \
--autoft_epochs 20 --inner_steps 5 \
--losses ce dcm entropy hinge flyp l1init l1zero l2init l2zero \
--ft_epochs 20 --lr 1e-5 --wd 0.2 --warmup_length 0 \
--repeats 50 --autoft_repeats 5
