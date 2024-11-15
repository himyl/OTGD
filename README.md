# OTGD

Code for ICONIP 2024 paper "Knowledge Distillation with Differentiable Optimal Transport on Graph Neural Networks"


![model](./framework.jpg)


## Installation
```
conda install --yes --file requirements.txt
```

## Running

1. Fetch the pretrained teacher models by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`

2. Run distillation by commands in `scripts\run_cifar_distill.sh`. An example of running OTGD is given by:

    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ceot --model_s resnet8x4 -a 1 -b 1 --mode hkd --trial 1
    ```

    where the flags are explained as:
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `None`
    - `-b`: the weight of other distillation losses, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.

## Citation

```
wait for camera ready
```

## Acknowledgement
This repo is based on the implementation of [CRD](https://github.com/HobbitLong/RepDistiller).
