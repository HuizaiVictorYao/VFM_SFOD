# Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection

This repository provides the implementation for our AAAI 2026 paper [**Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection**](https://arxiv.org/abs/2511.07301) .

## Paper
- Title: Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection
- Venue: AAAI 2026
- arXiv: https://arxiv.org/abs/2511.07301
- Abstract: Source-Free Object Detection (SFOD) aims to adapt a source-pretrained object detector to a target domain without access to source data. However, existing SFOD methods predominantly rely on internal knowledge from the source model, which limits their capacity to generalize across domains and often results in biased pseudo-labels, thereby hindering both transferability and discriminability. In contrast, Vision Foundation Models (VFMs), pretrained on massive and diverse data, exhibit strong perception capabilities and broad generalization, yet their potential remains largely untapped in the SFOD setting. In this paper, we propose a novel SFOD framework that leverages VFMs as external knowledge sources to jointly enhance feature alignment and label quality. Specifically, we design three VFM-based modules: (1) Patch-weighted Global Feature Alignment (PGFA) distills global features from VFMs using patch-similarity-based weighting to enhance global feature transferability; (2) Prototype-based Instance Feature Alignment (PIFA) performs instance-level contrastive learning guided by momentum-updated VFM prototypes; and (3) Dual-source Enhanced Pseudo-label Fusion (DEPF) fuses predictions from detection VFMs and teacher models via an entropy-aware strategy to yield more reliable supervision. Extensive experiments on six benchmarks demonstrate that our method achieves state-of-the-art SFOD performance, validating the effectiveness of integrating VFMs to simultaneously improve transferability and discriminability.

## Dataset Preparation
Please download datasets below and convert to COCO format (these datasets are all publicly available):

- Cityscapes: https://arxiv.org/abs/1604.01685
- Foggy Cityscapes: https://arxiv.org/abs/1708.07819
- Sim10k: https://arxiv.org/abs/1610.01983
- BDD100K: https://arxiv.org/abs/1805.04687 (after downloading, you may need to extract the "daytime" subset)
- KITTI: https://www.cvlibs.net/publications/Geiger2012CVPR.pdf
- ACDC: https://arxiv.org/abs/2104.13395

After download, convert annotations to COCO format if needed and organize the data like this:

```text
cityscapes/
  annotations/
    train.json
    val.json
  train/
    xxx
  val/
    xxx
foggy/
bdd_coco/
... (other dataset paths follow main_sfda.py:set_dataset_args)
```

For ACDC:

```text
acdc/
  gt_detection/
    fog/
    instancesonly_test_image_info.json
    instancesonly_train_gt_detection.json
    instancesonly_val_gt_detection.json
    night/
    rain/
    snow/
  rgb_anon/
    fog/
    night/
    rain/
    snow/
```

## Environment Setup
Python 3.10, CUDA >= 12.1

Create and activate a conda environment:

```bash
conda create -n vfm_sfod python=3.10 -y && conda activate vfm_sfod
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Compile Deformable DETR CUDA operators:

```bash
cd ./models/ops
sh ./make.sh
```

Install Grounding DINO:

```bash
pip install -e ./models/GroundingDINO
```

## Weights
1. Deformable DETR source weights:
   - Train on the source dataset following fundamentalvision/Deformable-DETR.
   - Put the trained source weights under the matching experiment directory, named `source.pth`, for example:
     `./exps/city2foggy_resnet50/source.pth`
   - Dataset paths are configured in `main_sfda.py:set_dataset_args` (you can modify them if needed). [Source-pretrained weights and target weights](https://drive.google.com/drive/folders/13WBIzQPXhlYH72v4szXrkMY6fT7srPWl?usp=sharing)

2. Grounding DINO weights:
   - Download the official weights and place them under `models/GroundingDINO/weights`, for example:
     `models/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth`


## Training

1. Source Pretraining: Please use the source-pretrained weights or follow [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) to train your source-pretrained weight and put it
in the corresponding directory, for example,  `./exps/city2foggy_resnet50/source.pth`

2. Source-free adaptation:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr_sfda.sh --backbone resnet50 --dataset_sfda [city_to_foggy,city_to_bdd, synthetic_to_real, city_to_rainy, kitti_to_city, city_to_acdc_fog, ...] --VFM_SFOD
```

