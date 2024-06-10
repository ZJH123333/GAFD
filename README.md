# GAFD
The source code for **G**raph **A**ugmentation Guided **F**ederated Knowledge **D**istillation (GAFD) framework.

## Paper
**Graph Augmentation Guided Federated Knowledge Distillation for Multisite Functional MRI Analysis and Brain Disorder Identification**

Qianqian Wang, Junhao Zhang, Lishan Qiao, Pew-Thian Yap, Mingxia Liu
Qianqian Wang and Junhao Zhang contributed equally to this work

## Datasets
We used the following datasets:

- ABIDE (Can be downloaded [here](http://fcon_1000.projects.nitrc.org/indi/abide/))
- REST-meta-MDD (Can be downloaded [here](http://rfmri.org/REST-meta-MDD))

## Dependencies
GAFD needs the following dependencies:

- python 3.9.18
- torch == 2.1.0
- torch_geometric == 2.4.0
- numpy == 1.24.1
- einops == 0.7.0
- sklearn == 1.3.2
- tqdm == 4.66.1
- pandas == 2.1.2
- h5py == 3.10.0

## Structure
    - `./main.py`: The main functions for GAFD.
    - `./load_ABIDE_data.py`: Data preparation for ABIDE.
    - `./load_MDD_data.py`: Data preparation for REST-meta-MDD.
    - `./model.py`: The model used in GAFD.
    - `./function.py`: The functions used in GAFD.
    - `./server.py`: This is federal aggregation function.
    - `./training_step.py`: This is used to perform model updates.
    - `./option.py`: This is used to adjust the options.
    - `./metrics.py`: This is used to calculate metrics.
