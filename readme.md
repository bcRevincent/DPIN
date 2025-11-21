# Experimental Dependency #
We provide our **pytorch** implementation for DPIN, and all dependencies of our code are listed in "requirements.txt".

## Machine Configurations ##
    Operating System: Win10 
    CPU: Intel Core i9-14700KF
    GPU: NVIDIA Geforce RTX 4090, 24G VRAM
    Memory: 128G
    Hard Disk: 4T SSD

## Code Structure ##
+ **config**: It contains the configuration of hyper-parameters for each dataset to reproduce our corresponding experimental results reported in the paper.
+ **dataset**: This folder contains three zero-shot dataset folders, a data pre-processing script ``dataset.py``, and a $N$-way $K$-shot sampler script ``Sampler.py``. The sources and brief descriptions of datasets are introduced in the paper, so you can download and save them into this folder to run our code. Especially, you can use ``vit_feature_extractor.py`` to extract the image features by the ViT-Base backbone.
+ **engine**: It consists of the training and evaluation script, and a main script. You can choose the running dataset in the ``main.py`` to reproduce the results. Especially, ``SUN_swa_vit_net.pt`` is the well-trained model can be evaluated directly for images on SUN.
+ **log**: This folder is used to record the training logs.
+ **models**: Our implementations of DPIN is included.
+ **tools**: Some separable modular implementations.

## A Well-trained Model ##
Because different experimental environments will affect the final model training results, we provide a model ``SUN_swa_vit_net.pt`` that has been trained on the most challenging SUN dataset to facilitate the reuse and reproduction of the experimental results.

## How to Run ##
We provide the optimal hyperparameter configurations for each dataset, and these can be found in the folder "./config". 
You can use the Python command to run ``./main.py`` directly with/without cmd arguments what you want to set. 
You can select one of the datasets to run and the details are described in the file ``./main.py``.
