# Implementation of Quantitative Testing with Concept Activation Vectors (TCAV)

### Introduction
This project implements the TCAV method that quantifies the contribution of specific image-concepts in deep network classifications: 

Paper : https://arxiv.org/abs/1711.11279

<img src="./tcav-vector.jpg">

### Dependencies
Python Version: **Python 3.7.1**

Package dependencies:

```numpy==1.23.2```
```torch==1.12.1```
```torchvision==0.13.1```
```scikit-learn==1.1.2```
```tqdm==4.64.1```
```matplotlib==3.5.3```

These can be installed by running ```pip install -r requirements.txt```

### Code structure

The code to run experiments is presented in the **main.ipynb** jupyter-notebook

The TCAV algorithim is implemented by the TCAV class in tcav.py

Other core-functionality can be found in the train.py, datasets.py and util.py python files in the base directory, implementing helper functions for model training, dataset/dataloader creation and utility functions respectively.

### Usage

1. Save the train and test image splits to the **data** folder in the base directory.
2. Save the concept example images (~30 samples) as a pickle file dump to the **data** folder in the base directory.

Execute the code in the **main.ipynb** jupyter-notebook to compute a cav vector, compute sensitivities, and to identify images from the train set that posess properties of the chosen image-concept.
