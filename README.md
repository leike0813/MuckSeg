# MuckSeg: A deep learning approach to real-time instance segmentation of TBM muck images

## Intruduction

MuckSeg is a deep learning approach to real-time instance segmentation of TBM muck images.

<img src="/docs/img1.jpg" alt="result1" width="256" height="512">
<img src="/docs/img2.jpg" alt="result1" width="256" height="512">
<img src="/docs/img3.jpg" alt="result1" width="256" height="512">

## Requirements

- pytorch 2.0.1 or above
- lightning 2.0.2 or above
- cuda support

## Dataset generation

Use the cli to generate training dataset from 2048×4096 original images:

```bash
python build_dataset.py --data-path <path-to-original-image-folder> --stages 1 2 3 --image-size 512 --num-repeats <stage1-repeat-time> <stage2-repeat-time>
```

## Train

Use the cli to train MuckSeg:

```bash
python train.py --cfg <path-to-config-file> --data-path <path-to-train-dataset>
```

Optionally, use the following command for fine-tuning:

```bash
python finetune.py --resume-from-run-path <path-to-last-run> --extra-cfg <path-to-finetune-config-file>
```

## Inference

Make batch inference by using the following command:

```bash
python inference.py --run-folder-path <path-to-run-folder> --inference-data-path <path-to-original-image-folder>
```

## Data availablity statement

If you wish to use the complete dataset for training MuckSeg, please contact zlzhou1@bjtu.edu.cn.