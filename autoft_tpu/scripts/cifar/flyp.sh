

python3 src/main.py --method flyp --model ViT-L/14 --data-location /iris/u/cchoi1/Data \
--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 1 --lr 3.75e-6 --wd 0.1 --batch-size 64 --warmup_length 4000 --workers 4 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitl14_openai_cifar10.pt