

python src/models/zeroshot.py --method autoft --model ViT-B/16 --data-location /iris/u/yoonho/data \
--id ImageNet --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet,ImageNetC \
--num_ood_hp_examples 1000 --use_class_balanced_ood --inner_steps 10 --autoft_epochs 100 --ft_epochs 10 \
--lr 1e-5 --wd 0.1 --batch-size 256 --warmup_length 1000 --accumulation_steps 2 \
--ft_data /iris/u/cchoi1/Data/ILSVRC2012/imagenet.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero \
--save ./zeroshot/clip_vitb16_imagenet2.pt --template openai_imagenet_template

python src/models/zeroshot.py --method autoft --model ViT-L/14 --data-location /iris/u/yoonho/data \
--id ImageNet --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet,ImageNetC \
--num_ood_hp_examples 1000 --use_class_balanced_ood --inner_steps 10 --autoft_epochs 100 --ft_epochs 10 \
--lr 1e-5 --wd 0.1 --batch-size 256 --warmup_length 1000 --accumulation_steps 2 \
--ft_data /iris/u/cchoi1/Data/ILSVRC2012/imagenet.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero \
--save ./zeroshot/clip_vitl14_imagenet2.pt --template openai_imagenet_template