# Description

This code corresponds to submission 6782: U-Flow: A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold.
It includes all needed code and the instructions to download the pretrained models, to reproduce the claimed results.
This code is being improved and documented, and will be published for re-utilization.

# Setup

Creating virtual environment and installing dependencies   
```bash
# Create conda virtual environment and activate it
conda create -n uflow python=3.10
conda activate uflow

# Install pytorch with cuda support
conda install -c conda-forge cudatoolkit=11.6
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# Install the rest of the dependencies with pip
pip install -r requirements.txt
 ```   

# Download data
To download MvTec AD dataset please enter the root directory and execute the following
```bash
cd <uflow-root-folder>/data
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
```

# [Optional] Download pre-trained models

If you are to reproduce the paper results, you could download the pre-trained models that were used to obtain the actual results, or you can even train a model with the provided code (explained in next section).
For downloading the pre-trained models, go to project root directory and inside there move to `models` folder.
```bash
cd <uflow-root-folder>/models
```

Download for auroc:
```bash
python -c 'import gdown; gdown.download_folder("https://drive.google.com/drive/folders/1vJaRnbe_q3LoWol35oSn5-uN1GacVsz0?usp=sharing", quiet=False)'
```

Download for iou:
```bash
python -c 'import gdown; gdown.download_folder("https://drive.google.com/drive/folders/18qGog2WbV7n6CzrvkL84R4d8Wtxo16Yx?usp=sharing", quiet=False)'
```

# Execution

There are three main files to execute: `train.py`, `predict.py`, and `evaluate.py`.
All scripts are to be run from the root directory `<uflow-root-folder>`.

You might need to add this folder to the pythonpath:
```bash
export PYTHONPATH=$PYTHONPATH:<uflow-root-folder>
```

## Train

For training





### Run
 Next, run it using `main.py`, and passing the image path. 

A test image is provided in `./images/test_image_01.jpg`

For example:
 ```bash
python main.py images/test_image_01.jpg
```

Other additional optional arguments:

| **Argument short name** | **Argument long name** |                                          **Description**                                           |     **Default value**     |
|:-----------------------:|:----------------------:|:--------------------------------------------------------------------------------------------------:|:-------------------------:|
|       image_path        |       image_path       |                                    Path of the image to process                                    | None (mandatory argument) |
|          -thr           |  --log_nfa_threshold   |                    Threshold over the computed NFA map, for final segmentation.                    |             0             |
|        -dist_thr        |  --distance_threshold  |       Threshold over the squared Mahalanobis distances, for computing the candidate regions.       |            0.5            |
|           -s            |         --size         |                          Input size for ResNet. Must be divisible by 32.                           |            256            |
|          -pca           |       --pca_std        | If float: the percentage of the variance to keep in PCA. If int: the number of components to keep. |            35             |
