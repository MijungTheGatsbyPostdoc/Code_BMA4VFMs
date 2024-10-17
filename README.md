# Code_BMA4VFMs

## Download data 
```bash
Use the links in [https://github.com/mlfoundations/model-soups/blob/main/datasets.md](https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/download_data.md)
```

## Convert raw data to feature representations under a selected CLIP model
```bash
Use [preprocess.py](https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/preprocess.py)
```

## Convert list to array
```bash
For ImageNet-1K, convert lists of feature representations into a large concatenated array of features using [reorganize_imagenet_feats.py](https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/reorganize_imagenet_feats.py). For OOD datasets of ImageNet, use [reorganize_imagenet_OOD_datasets_feats.py]
(https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/reorganize_imagenet_OOD_datasets_feats.py). For other datasets, this is not necessary.
```

## Train for MLE or MAP estimates 
```bash
[training_with_processed_data.py](https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/training_with_processed_data.py)
```

## Compute Hessian, then Compute Prediction Accuracy under BMA
```bash
[compute_Hessian.py](https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/compute_Hessian.py)
https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/predict_given_zeroshot_individually_trained_weights.py
```

## Compute posterior model weights without Hessian, then compute Prediction Accuracy under combinations of zeroshot and MAP (or MLE) models
```bash
[compute_Hessian_zeroshot_mle.py](https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/compute_Hessian_zeroshot_mle.py)
[combine_zeroshot_and_mle.py](https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/combine_zeroshot_and_mle.py)
```
