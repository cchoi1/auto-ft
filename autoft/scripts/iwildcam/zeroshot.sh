python3 src/models/zeroshot.py --method autoft --model ViT-L/14 --data-location /iris/u/cchoi1/Data \
--id IWildCamTrain --ood IWildCamOODVal --eval-datasets IWildCamIDVal,IWildCamIDTest,IWildCamOODTest \
--num_ood_hp_examples -1 --ft_epochs 20 \
--autoft_epochs 200 --inner_steps 10 --lr 1e-5 --wd 0.1 \
--batch-size 256 --warmup_length 500 --save ./zeroshot/clip_vitl14_iwildcam3.pt \
--losses ce flyp --template openai_imagenet_template