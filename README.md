
## Overview
This is the code for:

> Satrajit Chakrabarty, Pamela LaMontagne, Joshua Shimony, Daniel S. Marcus, Aristeidis Sotiras, "Non-invasive classification of IDH mutation status of gliomas from multi-modal MRI using a 3D convolutional neural network", in press.

It contains a 3D implementations of Mask R-CNN for classification of IDH mutation status of gliomas from volumetric MRI scans (T1c, T2, FLAIR).

## Installation
1. Setup package in a virtual environment:
```
git clone https://github.com/satrajitgithub/glioma_molecular_3D.git .
cd glioma_molecular_3D
virtualenv -p python3.6 venv
source venv/bin/activate
pip3 install -e .
```
2. Build CUDA functions using: `compile_cuda_scripts.sh` where the `-arch` parameter needs to be set according to the GPU used.


## Execution
1. Set I/O paths, model and training specifics in the configs file: `glioma_molecular_3D/config_files/idh/001.py`
2. Train the model:

    ```
    python exec.py --mode train -e 001 -d 3 -s idh
    ```
3. Run inference:
    ```
    python exec.py --mode train -e 001 -d 3 -s idh -m test
    ```
A pretrained model is provided at: `glioma_molecular_3D/pretrained_model/last_checkpoint/`

## License
This framework is published under the [Apache License Version 2.0](LICENSE).

## Acknowledgment
This repository is built on top of the [Medical Detection Toolkit](https://github.com/pfjaeger/medicaldetectiontoolkit.git).
