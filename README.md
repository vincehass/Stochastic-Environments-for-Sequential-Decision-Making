# Stochastic-Environments-for-Sequential-Decision-Making

This repo introduce Stochastic Generative Flow Networks, an innovative approach designed to model environments with inherent stochasticity in state transitions.

### Code references

Our implementation is heavily based on "Towards Understanding and Improving GFlowNet Training" (https://github.com/maxwshen/gflownet).

### Large Files

You can download additional large files by following link: https://drive.google.com/drive/folders/1JobUWGowoiQxGWVz3pipdfcjhtdY4CVq?usp=sharing

These files should be placed in `datasets`

### Installation

- Create conda environment:

```sh
conda create -n gfn-stochEnv python=3.10
conda activate gfn-stochEnv
```

## Environment Setup

To install dependecies, please run the command `pip install -r requirements.txt`.
Note that python version should be < 3.8 for running RNA-Binding tasks. You should install `pyg` with the following command

```
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

## Optional Dependencies

- Install PyTorch with CUDA. For our experiments we used the following versions:

```sh
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

or with pip see [most updated version](https://pytorch.org/get-started/locally/)

```sh
pip3 install torch torchvision torchaudio
```

You can change `pytorch-cuda=11.8` with `pytorch-cuda=XX.X` to match your version of `CUDA`.

- Install core dependencies:

```sh
pip install -r requirements.txt
```

-_(Optional)_ Install dependencies for molecule experiemtns

```sh
pip install -r requirements_mols.txt
```

### For MAC-OS USERS

Due to incompatibility of certain versions of `torch-sparse` and `torch-scatter`. You can run these commands

```sh
pip install git+https://github.com/rusty1s/pytorch_sparse.git
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html
pip --no-cache-dir install torch-geometric

```

## Running the code with all experiments

```sh
python grid/run_experiments.py
```

### Sequential Decision Making

```sh
python tfb/run_tfbind.py
```
