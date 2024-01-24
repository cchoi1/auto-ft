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

### Run Auto-FT

Sample scripts are in `auto-ft/scripts`:

- [Vanilla AutoFT](https://github.com/cchoi1/robust-optimizer/blob/master/autoft/scripts/iwildcam/autoft_is10_ex1000.sh)
- [Vanilla AutoFT with FLYP](https://github.com/cchoi1/robust-optimizer/blob/master/autoft/scripts/iwildcam/autoft_flyp_is10.sh)
- [Layerwise AutoFT](https://github.com/**cchoi1**/robust-optimizer/blob/master/autoft/scripts/iwildcam/autoft_layer_is50.sh)
- [Layerwise AutoFT with FLYP](https://github.com/cchoi1/robust-optimizer/blob/master/autoft/scripts/iwildcam/autoft_layer_is10_flyp.sh)


The `--inner_steps` and `--hopt_evals` parameters control the number of inner steps and outer steps respectively.
The `--num_ood_hp_examples` parameter controls the number of OOD val examples for hyperparameter optimization.
The effective batch size is controlled by `--batch-size` and `--accumulation_steps`. 
We use a batch size of 256 for all datasets except ImageNet, for which we use a batch size of 512.

To run Auto-FT with a FLYP loss term, use the following additional arguments:

```bash
--load PATH_TO_PRETRAINED_MODEL \
--ft_data DATA_DIR/csv/iwildcam_v2.0/iwildcam.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv
```

Sample command for running Auto-FT on ImageNet:

```
python src/main.py
    --method autoft
    --model ViT-B/16
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
