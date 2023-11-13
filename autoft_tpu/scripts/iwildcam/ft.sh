python3 src/finetune3.py --method autoft --model ViT-B/16 --data-location /home/carolinechoi/data/ \
--id IWildCamTrain --ood IWildCamOODVal --eval-datasets IWildCamIDVal,IWildCamIDTest,IWildCamOODTest \
--num_ood_hp_examples 1000 --use_class_balanced_ood --ft_epochs 20 --autoft_epochs 500 --inner_steps 100 \
--lr 1e-5 --wd 0.2 --batch-size 128 --warmup_length 500 --accumulation_steps 2 \
--load /home/carolinechoi/data/zeroshot/clip_vitb16_iwildcam.pt --template iwildcam_template \
--ft_data /home/carolinechoi/data/iwildcam_v2.0/iwildcam.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--losses ce \
--load_hparams /home/carolinechoi/data/hparams/iwildcam/vitb16_ce.json