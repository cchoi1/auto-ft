python3 src/models/zeroshot.py --method autoft --model ViT-B/16 --data-location /iris/u/cchoi1/Data \
--id StanfordCarsTrain --ood StanfordCarsValHOpt --eval-datasets StanfordCarsValEarlyStopping,StanfordCarsTest \
--num_ood_hp_examples 1000 --use_class_balanced_ood --ft_epochs 100 \
--autoft_epochs 500 --inner_steps 50 --lr 1e-5 --wd 0.0 \
--batch-size 128 --accumulation_steps 2 --warmup_length 500 \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero \
--ft_data /iris/u/cchoi1/Data/stanford-cars/train.csv --template stanfordcars_template \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--workers 2 --save ./zeroshot/clip_vitb16_flowers102.pt

python3 src/models/zeroshot.py --method autoft --model ViT-L/14 --data-location /iris/u/cchoi1/Data \
--id StanfordCarsTrain --ood StanfordCarsValHOpt --eval-datasets StanfordCarsValEarlyStopping,StanfordCarsTest \
--num_ood_hp_examples 1000 --use_class_balanced_ood --ft_epochs 100 \
--autoft_epochs 500 --inner_steps 50 --lr 1e-5 --wd 0.0 \
--batch-size 128 --accumulation_steps 2 --warmup_length 500 \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero \
--ft_data /iris/u/cchoi1/Data/stanford-cars/train.csv --template stanfordcars_template \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--workers 2 --save ./zeroshot/clip_vitl14_flowers102.pt