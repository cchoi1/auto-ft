#echo "========== CORE METHODS =========="
#echo "CIFAR AUTOFT WITH 1 INNER STEP SEED 0"
#python3 src/main.py --method autoft --model ViT-L/14 \
#--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
#--num_ood_hp_examples 100 --ft_epochs 10 \
#--autoft_epochs 100 --inner_steps 1 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
#--data-location /home/carolinechoi/robust-ft \
#--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
#--seed 0
#
#echo "CIFAR AUTOFT WITH 1 INNER STEP SEED 1"
#python3 src/main.py --method autoft --model ViT-L/14 \
#--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
#--num_ood_hp_examples 100 --ft_epochs 10 \
#--autoft_epochs 100 --inner_steps 1 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
#--data-location /home/carolinechoi/robust-ft \
#--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
#--seed 1
#
#echo "CIFAR AUTOFT WITH 10 INNER STEPS SEED 0"
#python3 src/main.py --method autoft --model ViT-L/14 \
#--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
#--num_ood_hp_examples 100 --ft_epochs 10 \
#--autoft_epochs 100 --inner_steps 10 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
#--data-location /home/carolinechoi/robust-ft \
#--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
#--load_hparams /home/carolinechoi/robust-ft/hparams/CIFAR10/is=10_evals=100_ex=100.json

#echo "CIFAR AUTOFT WITH 10 INNER STEPS SEED 1"
#python3 src/main.py --method autoft --model ViT-L/14 \
#--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
#--num_ood_hp_examples 100 --ft_epochs 10 \
#--autoft_epochs 100 --inner_steps 10 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
#--data-location /home/carolinechoi/robust-ft \
#--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
#--seed 1

echo "LAYERWISE CIFAR AUTOFT WITH 10 INNER STEPS SEED 0"
python3 src/main.py --method autoft --loss_type LayerwiseLoss --model ViT-L/14 \
--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 10 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
--seed 0

echo "LAYERWISE CIFAR AUTOFT WITH 10 INNER STEPS SEED 1"
python3 src/main.py --method autoft --loss_type LayerwiseLoss --model ViT-L/14 \
--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 10 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
--seed 1

echo "======= ABLATIONS ======="
echo "OOD = CIFAR10.2 SEED 0"
python3 src/main.py --method autoft --model ViT-L/14 \
--id CIFAR10 --ood CIFAR102 --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 1 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
--seed 0

echo "OOD = CIFAR10.2 SEED 1"
python3 src/main.py --method autoft --model ViT-L/14 \
--id CIFAR10 --ood CIFAR102 --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 1 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
--seed 1

echo "OOD = CIFAR10.1 SEED 0"
python3 src/main.py --method autoft --model ViT-L/14 \
--id CIFAR10 --ood CIFAR101 --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 1 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
--seed 0

echo "OOD = CIFAR10.1 SEED 1"
python3 src/main.py --method autoft --model ViT-L/14 \
--id CIFAR10 --ood CIFAR101 --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 1 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
--seed 1

echo "OOD = CIFAR10 SEED 0"
python3 src/main.py --method autoft --model ViT-L/14 \
--id CIFAR10 --ood CIFAR10 --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 1 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
--seed 0

echo "OOD = CIFAR10 SEED 1"
python3 src/main.py --method autoft --model ViT-L/14 \
--id CIFAR10 --ood CIFAR10 --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 1 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
--seed 1

echo "OOD = CINIC SEED 0"
python3 src/main.py --method autoft --model ViT-L/14 \
--id CIFAR10 --ood CINIC --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 1 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
--seed 0

echo "OOD = CINIC SEED 1"
python3 src/main.py --method autoft --model ViT-L/14 \
--id CIFAR10 --ood CINIC --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 1 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
--seed 1

#echo "ID VAL THRESHOLD ON CIFAR SEED 2"
#python3 src/main.py --method autoft --model ViT-L/14 \
#--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
#--num_ood_hp_examples 100 --ft_epochs 10 \
#--autoft_epochs 100 --inner_steps 10 --lr 3e-5 --wd 0.1 --batch-size 32 --accumulation_steps 8 --warmup_length 1000 \
#--data-location /home/carolinechoi/robust-ft \
#--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
#--load_hparams /home/carolinechoi/robust-ft/hparams/CIFAR10/is=10_evals=100_ex=100_idval.json
