# AutoFT: Robust Fine-tuning via Hyperparameter Optimization on OOD Data

This repository contains code for the paper [AutoFT: Robust Fine-tuning via Hyperparameter Optimization on OOD Data](https://arxiv.org/abs/2401.10220) by Caroline Choi*, Yoonho Lee*, Annie Chen, Allan Zhou, Aditi Raghunathan, and Chelsea Finn.

Our code is heavily based on [WiSE-FT](https://github.com/mlfoundations/wise-ft), [Open-CLIP](https://github.com/mlfoundations/open_clip), and [FLYP](https://github.com/locuslab/FLYP). We thank the authors for open-sourcing their code.


### Install dependencies

```
cd path/to/your/project
python -m venv auto-ft
source auto-ft/bin/activate
pip install -r requirements.txt
```

### Add directory to PYTHONPATH:

```
cd path/to/your/project
cd auto-ft
export PYTHONPATH="$PYTHONPATH:$PWD"
```

### Datasets

To download datasets, please refer to these [these instructions](https://github.com/mlfoundations/wise-ft/blob/master/datasets.md) and follow the dataset directory structures described [here](https://github.com/locuslab/FLYP/blob/main/DATA.md).

### Generating zero-shot CLIP models

Run src/models/zeroshot.py with the additional arguments `--template` and `--save`:
```
python src/models/zeroshot.py \
    --method autoft \
    --model ViT-B/16 \
    --data-location DATA_DIR \
    --id ImageNet \
    --ood ImageNetC \
    --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet,ImageNetC \
    --ft_data DATA_DIR/csv/imagenet.csv \
    --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
    --save ./zeroshot/clip_vitb16_imagenet.pt \
    --template openai_imagenet_template
```

### Run Auto-FT

Sample scripts are in `auto-ft/scripts`:

The `--inner_steps` and `--hopt_evals` arguments specify the number of inner steps and outer loop evaluations respectively.
The `--num_ood_hp_examples` argument specifies the number of OOD val examples for hyperparameter optimization.
Effective batch size is controlled by `--batch-size` and `--accumulation_steps`. 
We use a batch size of 256 for all datasets except ImageNet, for which we use a batch size of 512.

To run Auto-FT with a FLYP loss term, use the following additional arguments:

```bash
--load PATH_TO_PRETRAINED_MODEL \
--ft_data DATA_DIR/csv/iwildcam_v2.0/iwildcam.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv
```
To run Auto-FT with a layerwise loss and/or a layerwise learning rates and weight decays, use the arguments `--layerwise_loss`, `--layerwise_opt`.

Sample command for running Auto-FT on ImageNet:

```
python src/main.py \
    --method autoft \
    --model ViT-B/16 \
    --data-location DATA_DIR \
    --id ImageNet \
    --ood ImageNetC \
    --eval-datasets ImageNet,ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet \
    --num_ood_hp_examples 1000 \
    --inner_steps 5 \
    --hopt_evals 100 \
    --lr 1e-5 \
    --wd 0.1 \
    --batch-size 512 \
    --warmup_length 500 \
    --ft_epochs 10 \
    --load /path/to/zeroshot/clip/model \
    --ft_data DATA_DIR/csv/imagenet.csv \
    --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
    --losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero \
    --template openai_imagenet_template
```

Sample command for running Auto-FT on iWildCam:
```
python src/main.py \
    --method autoft \
    --model ViT-B/16 \
    --data-location DATA_DIR \
    --id IWildCamTrain \
    --id_val IWildCamIDVal \
    --ood IWildCamOODVal \
    --eval-datasets IWildCamIDVal,IWildCamIDTest,IWildCamOODTest \
    --num_ood_hp_examples 1000 \
    --use_class_balanced_ood \
    --ft_epochs 20 \
    --hopt_evals 500 \
    --inner_steps 100 \
    --lr 1e-5 --wd 0.2 --batch-size 128 --warmup_length 500 --accumulation_steps 2 \
    --load PATH_TO_ZEROSHOT_MODEL \
    --template iwildcam_template \
    --ft_data DATA_DIR/csv/iwildcam_v2.0/iwildcam.csv \
    --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
    --losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero \
    --clip_gradient
```

Sample command for running Auto-FT on FMoW:
```
python src/main.py \
    --method autoft \
    --model ViT-B/16 \
    --data-location DATA_DIR \
    --id FMOWTrain \
    --ood FMOWOODVal \
    --eval-datasets FMOWIDVal,FMOWIDTest,FMOWOODTest \
    --num_ood_hp_examples 620 --use_class_balanced_ood \
    --ft_epochs 20 \
    --hopt_evals 500 \
    --inner_steps 100 \
    --lr 1e-5 --wd 0.1 \
    --batch-size 128 --accumulation_steps 2 --warmup_length 500 \
    --losses ce dcm flyp entropy hinge l1init l1zero l2init l2zero \
    --ft_data DATA_DIR/csv/fmow_v1.1/fmow.csv \
    --template fmow_template \
    --csv-img-key filepath --csv-caption-key title --get_labeled_csv \
    --load PATH_TO_ZEROSHOT_MODEL \
    --clip_gradient
```


## Citing

If you found this repository useful, please consider citing:
```bibtex
@article{choi2024autoft,
  title={AutoFT: Robust Fine-Tuning by Optimizing Hyperparameters on OOD Data},
  author={Choi, Caroline and Lee, Yoonho and Chen, Annie and Zhou, Allan and Raghunathan, Aditi and Finn, Chelsea},
  journal={arXiv preprint arXiv:2401.10220},
  year={2024}
}
```
