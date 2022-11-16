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
tar -xvf mvtec_anomaly_detection.tar.xz
rm -xvf mvtec_anomaly_detection.tar.xz
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

For training, the only command line argument required is the category:
```bash
usage: train.py [-h] -cat CATEGORY [-config CONFIG_PATH] [-data DATA] [-train_dir TRAINING_DIR]
```

A basic execution could be for example:
```bash
python src/train.py -cat carpet
```

Command line arguments:

| **Argument short name** | **Argument long name** |                                                                        **Description**                                                                        |                             **Default value**                             |
|:-----------------------:|:----------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------:|
|          -cat           |       --category       | MvTec category to train. One of [carpet, grid, leather, tile, wood, bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper] |                         None (mandatory argument)                         |
|         -config         |     --config_path      |                                       Config file path. If Not specified, uses the default config in `configs` folder.                                        | None: loads the config in `configs` folder for the corresponding category |
|          -data          |         --data         |                                 Folder with MvTec AD dataset. Inside this folder there must be one folder for each category.                                  |                          uflow-root-folder/data                           |
|       -train_dir        |     --training_dir     |                                                             Folder to save training experiments.                                                              |                        uflow-root-folder/training                         |


The script will generate logs inside `<uflow-root-folder>/training` folder (or a different one if you changed it with the command line arguments), and will log metrics and images to tensorboard.

Tensorboard can be executed as:
```bash
cd <uflow-root-folder>/training
tensorboard --logdir .
```

## Predict
This script performs the inference image by image for the chosen category and displays the results.

```bash
usage: predict.py [-h] -cat CATEGORY [-data DATA]
```

| **Argument short name** | **Argument long name** |                                                                        **Description**                                                                        |   **Default value**    |
|:-----------------------:|:----------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------:|
|          -cat           |       --category       | MvTec category to train. One of [carpet, grid, leather, tile, wood, bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper] |         carpet         |
|          -data          |         --data         |                                 Folder with MvTec AD dataset. Inside this folder there must be one folder for each category.                                  | uflow-root-folder/data |

For example use like this:
```bash
python src/predict.py -cat carpet
```

## Evaluate
This script run the inference and evaluates auroc and segmentation iou, for reproducing results.

```bash
usage: evaluate.py [-h] -cat CATEGORIES [CATEGORIES ...] [-data DATA]
                   [-hp HIGH_PRECISION]
```

| **Argument short name** | **Argument long name** |                                                                            **Description**                                                                            |            **Default value**             |
|:-----------------------:|:----------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------:|
|          -cat           |       --category       | MvTec categories to train. A subset of [carpet, grid, leather, tile, wood, bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper]. | None: meaning to run over all categories |
|          -data          |         --data         |                                     Folder with MvTec AD dataset. Inside this folder there must be one folder for each category.                                      |          uflow-root-folder/data          |
|           -hp           |    --high-precision    |         Whether to use high precision for computing the NFA values or not. High precision acieves slightly better performance but takes more time to execute.         |                  False                   | 

Example usage for two categories:
```bash
python src/evaluate -cat carpet grid
```
