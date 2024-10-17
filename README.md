# Code_BMA4VFMs

### Download data 
```bash
https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/download_data.md
```

### Convert raw data to feature representations under a selected CLIP model
```bash
https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/preprocess.py
```

### Convert list to array
For ImageNet-1K, convert lists of feature representations into a large concatenated array of features using
```bash
https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/reorganize_imagenet_feats.py
```
For OOD datasets of ImageNet, use
```bash
https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/reorganize_imagenet_OOD_datasets_feats.py
```
For other datasets, this is not necessary.

### Train for MLE or MAP estimates 
```bash
https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/training_with_processed_data.py
```

### Compute Hessian, then Compute Prediction Accuracy under BMA
First, compute Hessian using
```bash
https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/compute_Hessian.py
```
Then, compute prediction accuracy using 
```bash
https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/predict_given_zeroshot_individually_trained_weights.py
```

### Compute posterior model weights without Hessian, then compute Prediction Accuracy under combinations of zeroshot and MAP (or MLE) models
First, compute posterior model weights using
```bash
https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/compute_Hessian_zeroshot_mle.py
```
Then, compute prediction accuracy using 
```bash
https://github.com/MijungTheGatsbyPostdoc/Code_BMA4VFMs/blob/main/combine_zeroshot_and_mle.py
```
