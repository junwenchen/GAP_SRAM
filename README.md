# GAP_SRAM
## Introduction
This repo is the PyTorch implementation of ([Group Activity Prediction with Sequential Relational Anticipation Model](https://arxiv.org/pdf/2008.02441.pdf)) in ECCV 2020.
        
        


## Dependencies

- Python `3.x`
- PyTorch `1.0.1`
- numpy, pickle, scikit-image
- [RoIAlign for Pytorch](https://github.com/longcw/RoIAlign.pytorch)
- Datasets: [Volleyball](https://github.com/mostafa-saad/deep-activity-rec)




## Prepare Datasets

1. Download [volleyball](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip).
2. Unzip the dataset file into `data/volleyball`.


## Get Started
    ```shell
    python scripts/train_volleyball.py
    ```
    
    You can specify the running arguments in the python files under `scripts/` directory. The meanings of arguments can be found in `config.py`

Note that part of codes are referred from ([ARG](https://github.com/wjchaoGit/Group-Activity-Recognition)) project.

