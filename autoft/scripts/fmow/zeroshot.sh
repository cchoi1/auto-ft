python3 src/models/zeroshot.py --method autoft --model ViT-L/14 --data-location /iris/u/yoonho/data/wilds \
--id FMOWTrain --ood FMOWOODVal --eval-datasets FMOWIDVal,FMOWIDTest,FMOWOODTest \
--num_ood_hp_examples 1000 --ft_epochs 20 \
--autoft_epochs 500 --inner_steps 10 --lr 1e-5 --wd 0.1 \
--batch-size 256 --warmup_length 500 \
--template fmow_template --save ./zeroshot/clip_vitl14_fmow2.pt

python3 src/models/zeroshot.py --method autoft --model ViT-L/14@336px --data-location /iris/u/yoonho/data/wilds \
--id FMOWTrain --ood FMOWOODVal --eval-datasets FMOWIDVal,FMOWIDTest,FMOWOODTest \
--num_ood_hp_examples 1000 --ft_epochs 20 \
--autoft_epochs 500 --inner_steps 10 --lr 1e-5 --wd 0.1 \
--batch-size 256 --warmup_length 500 \
--template fmow_template --save ./zeroshot/clip_vitl14_fmow336px.pt