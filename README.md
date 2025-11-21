# PROGRESSOR: A Perceptually Guided Reward Estimator with Self-Supervised Online Refinement

<p align="center">
   <a href="https://ripl.github.io/progressor/">[Project Website]</a>&emsp;<a href="https://arxiv.org/abs/2411.17764">[arXiv]</a>&emsp;<a href="https://openaccess.thecvf.com/content/ICCV2025/html/Ayalew_PROGRESSOR_A_Perceptually_Guided_Reward_Estimator_with_Self-Supervised_Online_Refinement_ICCV_2025_paper.html">[ICCV 2025 Paper]</a>
</p>

This repository contains the official implementation of **PROGRESSOR**, a novel framework that learns a task-agnostic reward function from videos, enabling policy training through goal-conditioned reinforcement learning (RL) without manual supervision. PROGRESSOR estimates the distribution over task progress as a function of the current, initial, and goal observations, learned in a self-supervised fashion. The framework includes adversarial online refinement to mitigate distribution shift during RL training.

## Overview

PROGRESSOR consists of two main components:

1. **Video2Reward**: Self-supervised reward model training from video demonstrations
2. **DrM**: Reinforcement learning agent that uses the learned reward function for policy training

The reward model is pretrained on large-scale egocentric human video from EPIC-KITCHENS and can generalize to real-robot offline RL under noisy demonstrations without fine-tuning on in-domain task-specific data.

## Repository Structure

```
PROGRESSOR/
├── Video2Reward/          # Reward model training code
│   ├── main.py            # Main training script
│   ├── main_ours.py       # Alternative training script
│   ├── reward_model.py    # Reward model implementation
│   ├── simple_reward_model.py
│   └── util/              # Utilities and data loaders
│       ├── frame_triplet_loader.py
│       ├── transforms.py
│       └── util.py
├── DrM/                   # RL training code (based on DrM)
│   ├── train_mw.py        # MetaWorld training script
│   ├── metaworld_env.py   # MetaWorld environment wrapper
│   ├── agents/            # RL agent implementations
│   ├── cfgs/              # Configuration files
│   └── pretrained_reward/ # Pretrained reward models
└── include/               # OpenGL headers
```

## Installation

### Prerequisites

- Python 3.8+ (3.10.12 recommended for Video2Reward)
- CUDA-capable GPU
- Linux (tested on Ubuntu)

### Step 1: Install Video2Reward Dependencies

For training the reward model:

```bash
cd Video2Reward
python3 -m venv env
source env/bin/activate

# Add the project to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install DrM Dependencies

For RL training with MetaWorld:

```bash
cd DrM

# Install system dependencies
sudo apt update
sudo apt install libosmesa6-dev libegl1-mesa libgl1-mesa-glx libglfw3

# Create conda environment
conda env create -f conda_env.yml
conda activate drm

# Install PyTorch (adjust CUDA version as needed)
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install MetaWorld
cd metaworld
pip install -e .
cd ..

# Install additional dependencies
cd rrl-dependencies
pip install -e .
cd mj_envs
pip install -e .
cd ..
cd mjrl
pip install -e .
cd ../..
```

**Note**: Ensure that `mujoco_py` can use GPU rendering for better performance during training.

## Pretrained Weights

Pretrained weights for PROGRESSOR are available on Hugging Face:

- **Download**: [PROGRESSOR-pretrained-weight](https://huggingface.co/tedywond/PROGRESSOR-pretrained-weight)

Available models:
- **`PROGRESSOR_EPIC_KITCHENS_pretrained_weight.pth`**: Pretrained on EPIC-KITCHENS dataset for general video-to-reward learning
- **`all_tasks.pth`**: Pretrained model for MetaWorld tasks that we used for our evaluations, ready to use for RL training

**To replicate our experiments:** Download `all_tasks.pth` from the Hugging Face repository and place it in the `DrM/pretrained_reward/` folder. The code will automatically load this pretrained model when training on MetaWorld tasks, avoiding the need for pretraining from scratch.

You can download the pretrained weights and use them directly for inference or as initialization for fine-tuning on your specific tasks.

## Expert Demonstrations

Expert demonstrations used for pretraining the MetaWorld reward model are available:

- **Dataset**: [PROGRESSOR_metaworld_expert_demonstrations](https://huggingface.co/datasets/tedywond/PROGRESSOR_metaworld_expert_demonstrations)

This dataset contains the expert demonstrations used to pretrain the `all_tasks.pth` model for MetaWorld tasks.

## Usage

### Training RL Policies with PROGRESSOR

The main way to run PROGRESSOR is through `train_mw.py`, which handles reward model pretraining automatically:

```bash
cd DrM

# Main training command (handles pretraining automatically)
python train_mw.py task=$1 agent=drqv2 seed=$2

# Example usage:
python train_mw.py task=drawer-open agent=drqv2 seed=0
python train_mw.py task=hammer agent=drqv2 seed=42

# Available tasks: drawer-open, door-open, hammer, peg-insert-side, pick-place, reach
# See cfgs/task/ for all available task configurations
```

**Important:** Before running training, you need to set the `exp_data_path` configuration. You can either:

1. **Edit the config file**: Update `exp_data_path` in `DrM/cfgs/config.yaml` (line 6) with your expert data path
2. **Override via command line**: Add `exp_data_path=/path/to/your/expert/data` to your training command:
   ```bash
   python train_mw.py task=drawer-open agent=drqv2 seed=0 exp_data_path=/path/to/your/expert/data
   ```

The expert demonstrations can be downloaded from the [PROGRESSOR_metaworld_expert_demonstrations](https://huggingface.co/datasets/tedywond/PROGRESSOR_metaworld_expert_demonstrations) dataset.

**Note:** To use the pretrained reward model (recommended for replicating our results), download `all_tasks.pth` from the [pretrained weights repository](https://huggingface.co/tedywond/PROGRESSOR-pretrained-weight) and place it in `DrM/pretrained_reward/all_tasks.pth`. If the file is not found, the code will automatically pretrain the model from expert demonstrations, which takes additional time.

**Key Configuration Options:**
- `task`: MetaWorld task name
- `agent`: RL agent (drqv2 for standard training, drm_mw for DrM agent)
- `seed`: Random seed for reproducibility
- `reward_model`: Reward model type (ours for PROGRESSOR, set automatically)
- `online_finetune`: Enable adversarial online refinement (default: True)
- `exp_data_path`: Path to expert demonstration data (required for pretraining and online finetuning)

<details>
<summary><b>Training the Reward Model Separately (Optional)</b></summary>

If you want to train the reward model separately before RL training:

```bash
cd Video2Reward

# Basic training
python main_ours.py \
    --data_path /path/to/your/video/data \
    --output_dir ./output \
    --batch_size 8 \
    --epochs 100 \
    --normalize_prediction \
    --randomize \
    --augment

# Distributed training
python -m torch.distributed.launch --nproc_per_node=4 main_ours.py \
    --data_path /path/to/your/video/data \
    --output_dir ./output \
    --distributed
```

**Key Arguments:**
- `--data_path`: Path to video dataset (should contain task folders with train/test splits)
- `--output_dir`: Directory to save checkpoints and outputs
- `--normalize_prediction`: Normalize trajectory progress to [0, 1]
- `--randomize`: Use randomized frame sampling
- `--augment`: Apply data augmentation

**Note:** Before running the training scripts, you need to update the placeholder paths in the code:
- In `Video2Reward/main_ours.py` at line 262, replace `root_dir='/path/to/data'` with your actual data path
- In `Video2Reward/main.py` at line 252, replace `root_dir='/path/to/data'` with your actual data path

</details>

### Data Format

The reward model expects data organized as:
```
data_path/
├── task1/
│   ├── train/
│   │   ├── 0/
│   │   │   ├── 0.png
│   │   │   ├── 1.png
│   │   │   └── ...
│   │   └── ...
│   └── test/
│       └── ...
└── task2/
    └── ...
```

Each trajectory folder contains sequential frames (0.png, 1.png, ..., 100.png) representing a complete task execution.

## Method Details

### Self-Supervised Reward Learning

PROGRESSOR learns a reward model that predicts the distribution of progress:

$$E_{\theta}(\mathcal{o}_i, \mathcal{o}_j, \mathcal{o}_g) = \mathcal{N}(\mu, \sigma^2)$$

The reward function is derived as:

$$r_{\theta}(\mathcal{o}_i, \mathcal{o}_j, \mathcal{o}_g) = \mu - \alpha \mathcal{H}(\mathcal{N}(\mu, \sigma^2))$$

where $\mu$ is the predicted mean progress and $\mathcal{H}$ is the entropy.

### Adversarial Online Refinement

During online RL training, PROGRESSOR uses a "push-back" strategy to handle distribution shift. For out-of-distribution observations from non-expert trajectories, the model learns to push back predictions by a decay factor $\beta \in [0,1]$:

$$\mu_{\tau_k'} \leftarrow \beta \mu_{\tau_k'}$$

This enables the reward model to differentiate between in-distribution expert data and out-of-distribution exploration data.

## Experiments

### Simulated Experiments

PROGRESSOR is evaluated on six MetaWorld manipulation tasks:
- door-open
- drawer-open  
- hammer
- peg-insert-side
- pick-place
- reach

### Real-World Experiments

The model is pretrained on EPIC-KITCHENS and evaluated on real-robot tasks without fine-tuning, demonstrating strong zero-shot generalization.

## Citation

If you use PROGRESSOR in your research, please cite:

```bibtex
@InProceedings{Ayalew_2025_ICCV,
  author = {Ayalew, Tewodros W. and Zhang, Xiao and Wu, Kevin Yuanbo and Jiang, Tianchong and Maire, Michael and Walter, Matthew R.},
  title = {PROGRESSOR: A Perceptually Guided Reward Estimator with Self-Supervised Online Refinement},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2025},
  pages = {10297-10306}
}
```

## Acknowledgments

DrM is licensed under the MIT license. MuJoCo and DeepMind Control Suite are licensed under the Apache 2.0 license. We would like to thank the DrM authors for open-sourcing the [DrM](https://github.com/facebookresearch/drm) codebase and for implementing the integration of DrQ-v2 with MetaWorld environment, which we extended for our work. Our RL implementation builds on top of their repository.

We also acknowledge:
- [Meta-World](https://meta-world.github.io/) project for providing the manipulation environments
- EPIC-KITCHENS dataset for pretraining the reward model
- [DrQ-v2](https://github.com/facebookresearch/drqv2) authors, whose codebase DrM builds upon

## License

Please refer to the individual component licenses:
- Video2Reward: See Video2Reward/LICENSE
- DrM: See DrM/LICENSE

## Contact

For questions and issues, please refer to the [project website](https://ripl.github.io/progressor/).

