python3 src/finetune3.py --method autoft --model ViT-B/16 \
--data-location /home/carolinechoi/data \
--id ImageNet4 --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet,ImageNetC \
--num_ood_hp_examples 15000 --use_class_balanced_ood --inner_steps 10 --autoft_epochs 1000 --ft_epochs 10 \
--lr 1e-5 --wd 0.1 --batch-size 128 --warmup_length 1000 --accumulation_steps 4 \
--load /home/carolinechoi/data/zeroshot/clip_vitb16_imagenet2.pt \
--ft_data /iris/u/cchoi1/Data/csv/imagenet.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero --template openai_imagenet_template \
--load_hparams /home/carolinechoi/hparams/imagenet4/is10_evals1000_ex15000.json
