python3 src/models/zeroshot.py --method autoft --model ViT-L/14 --data-location /home/carolinechoi/data \
--id IWildCamTrain --ood IWildCamOODVal --eval-datasets IWildCamIDVal,IWildCamIDTest,IWildCamOODTest \
--num_ood_hp_examples 1000 --ft_epochs 20 \
--autoft_epochs 500 --inner_steps 10 --lr 1e-5 --wd 0.1 \
--batch-size 256 --warmup_length 500 \
--template fmow_template --save /home/carolinechoi/data/zeroshot/clip_vitl14_iwildcam.pt \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero

python3 src/models/zeroshot.py --method autoft --model ViT-B/16 --data-location /home/carolinechoi/data \
--id IWildCamTrain --ood IWildCamOODVal --eval-datasets IWildCamIDVal,IWildCamIDTest,IWildCamOODTest \
--num_ood_hp_examples 1000 --ft_epochs 20 \
--autoft_epochs 500 --inner_steps 10 --lr 1e-5 --wd 0.1 \
--batch-size 256 --warmup_length 500 \
--template fmow_template --save /home/carolinechoi/data/zeroshot/clip_vitb16_iwildcam.pt \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero