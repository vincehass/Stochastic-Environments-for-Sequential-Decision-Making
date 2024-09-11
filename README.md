# Stochastic-Environments-for-Sequential-Decision-Making

This repo introduce Stochastic Generative Flow Networks, an innovative approach designed to model environments with inherent stochasticity in state transitions.

#Implementation

## Environment Setup

To install dependecies, please run the command `pip install -r requirement.txt`.
Note that python version should be < 3.8 for running RNA-Binding tasks. You should install `pyg` with the following command

```
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

### Code references

Our implementation is heavily based on "Towards Understanding and Improving GFlowNet Training" (https://github.com/maxwshen/gflownet).

### Large Files

You can download additional large files by following link: https://drive.google.com/drive/folders/1JobUWGowoiQxGWVz3pipdfcjhtdY4CVq?usp=sharing

These files should be placed in `datasets`
