# *MultiHem*: Multitask Learning on Segmentation and Classification of Intracranial Hemorrhage Using CT Scans

> [!WARNING]
> You should not use this code for other projects without the permission of the authors.

## Install Environments (tested on Ubuntu 22.04)

```bash
git clone https://github.com/mikami520/MultiHem.git
cd MultiHem
conda-env create -f environment.yml
conda acticate hem
```

## Preprocessing and Data Augmentation

All the preprocessing and data augmentation are done through `monai.transform` which can be checked in the `trainer.py`

## Training

Before training, modify the `config.yml` to predefine what you want to do (dataset path, network architecture, hyper-parameters, etc)

```bash
python main.py --cfg config.yml --device cuda
```

## Inference

```bash
python main.py --cfg config.yml --device cuda --test
```

