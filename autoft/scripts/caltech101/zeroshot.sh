python3 src/models/zeroshot.py --method autoft --model ViT-B/16 --data-location /iris/u/cchoi1/Data \
--id Caltech101Train --ood Caltech101ValHOpt --eval-datasets Caltech101ValEarlyStopping,Caltech101Test \
--num_ood_hp_examples 1000 --ft_epochs 20 \
--autoft_epochs 500 --inner_steps 10 --lr 1e-5 --wd 0.1 \
--batch-size 256 --warmup_length 500 \
--template caltech101_template --save ./zeroshot/clip_vitb16_caltech101.pt --losses ce flyp