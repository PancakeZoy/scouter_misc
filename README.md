# scouter_misc

This Repo contains the code to reproduce the results and figures in manuscript.

Official `Scouter` package repository: [Link](https://github.com/PancakeZoy/scouter).

## Folder Organization

| Name | Content |
|-----------------|-------------|
| [data](data) | The datasets for the training of `GEARS`, `biolord`, and `Scouter`, respectively|
| [code](code) | The folder of codes for the training and evaluation|
| [results](results) | The folder that stores the results from model training and evaluation|

To run the training scripts, please install the following packages:
```
pip install scouter-learn
pip install cell-gears
pip install biolord
```