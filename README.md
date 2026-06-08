## Overview

SPGCRL combines graph contrastive learning (GCL) with Double DQN (DDQN) to solve the influence maximization problem on directed social networks. The framework consists of three stages:

1. **Strategy 1 (DataStrategy1.py)** — Path inverse entropy H(v) based subgraph augmentation
2. **Strategy 2 (DataStrategy2.py)** — Propagation control C(i) based subgraph augmentation  
3. **GCL Pre-training (GCL.py)** — Contrastive learning on augmented graph views
4. **DDQN Seed Selection (DDQN.py)** — Reinforcement learning for influence maximization

## Project Structure

```
SPGCRL/
├── data/                  # Graph datasets (included)
├── DataStrategy1.py       # Strategy 1: HV-based augmentation & HV Predictor training
├── DataStrategy2.py       # Strategy 2: Propagation control augmentation & C Predictor training
├── GCL.py                 # Graph contrastive learning model & training
├── DDQN.py                # Double DQN agent for seed selection
├── ic3.py                 # Information cascade propagation model (IC3)
├── data_loader.py         # Dataset loading utilities
├── weight_calculate.py    # Edge weight computation
└── requirements.txt
```

## Installation

### 1. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions, see [PyTorch Get Started](https://pytorch.org/get-started/locally/).

### 2. Install PyG and extensions

```bash
pip install torch-geometric==2.6.1
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train HV Predictor (Strategy 1)

```bash
python DataStrategy1.py --dataset_name soc_epinions
```

### Step 2: Train C Predictor (Strategy 2)

```bash
python DataStrategy2.py --dataset_name soc_epinions
```

### Step 3: Train GCL Model

```bash
python GCL.py --dataset_name soc_epinions \
    --hv_weights_path checkpoints/HV_save/soc_epinions_hv_predictor_best.pth \
    --c_weights_path checkpoints/C_save/soc_epinions_c_predictor_best.pth
```

### Step 4: Run DDQN for Influence Maximization

```bash
python DDQN.py --dataset_name soc_epinions \
    --gcl_model_path gcl_models/soc_epinions_best_gcl_model.pth
```



## Available Datasets

| Identifier       | Description                          |
|------------------|--------------------------------------|
| petster_hamster  | Petster hamster social network       |
| fb_tvshow        | Facebook TV show page network        |
| fb_politician    | Facebook politician page network     |
| soc_advogato     | Advogato social network              |
| fb_public        | Facebook public figure page network  |
| soc_epinions     | Epinions social trust network        |


 6 datasets are included in the `data/` directory. 
