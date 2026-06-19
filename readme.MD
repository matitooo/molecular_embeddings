# Bacteria grotwh inhibition prediction using GNN-based molecules embeddings

## Individual Research Module Project, SoSe26, Potsdam Universität

Using [Trimnet ](https://github.com/yvquanli/TrimNet) and [3DInfomax](https://github.com/hannesstark/3dinfomax), alongside with a graph-based embedding layer, different GNN-based molecule representation layers are evaluated to assess whether Morgan Fingerprints can be replaced in drug prediction tasks.

Training and evaluation were performed over the [DropArray](https://www.pnas.org/doi/10.1073/pnas.2402017122) dataset.

## How to use

### 1. Environment setup

A stable, python 3.11-based conda environment can be set up by running the following command from inside the repo root folder

```
conda create -f environment_config_stable.yaml
```
