# PrefMMT
This repository contains the source code for our paper: "PrefMMT: Modeling Human Preferences in Preference-based Reinforcement Learning with Multimodal Transformers", submitted to 2025 IEEE International Conference on Robotics and Automation (ICRA 2025). For more details, please refer to [our project website](https://sites.google.com/view/prefmmt).

## Abstract
Preference-based reinforcement learning (PbRL) shows promise in aligning robot behaviors with human preferences, but its success depends heavily on the accurate modeling of human preferences through reward models. Most methods adopt Markovian assumptions for preference modeling (PM), which overlook the temporal dependencies within robot behavior trajectories that impact human evaluations. While recent works have utilized sequence modeling to mitigate this by learning sequential non-Markovian rewards, they ignore the multimodal nature of robot trajectories, which consist of elements from two distinctive modalities: state and action. As a result, they often struggle to capture the complex interplay between these modalities that significantly shapes human preferences. In this paper, we propose a multimodal sequence modeling approach for PM by disentangling state and action modalities. We introduce a multimodal transformer network, named PrefMMT, which hierarchically leverages intra-modal temporal dependencies and inter-modal state-action interactions to capture complex preference patterns. We demonstrate that PrefMMT consistently outperforms state-of-the-art PM baselines on locomotion tasks from the D4RL benchmark and manipulation tasks from the Meta-World benchmark.

## Overview Architecture for PrefMMT
<div align=center>
<img src="/figures/Comparison.jpg" width="800" />
</div> 

## Usage
### Requirements
1. Install dependencies

```
pip install --upgrade pip
conda install -y -c conda-forge cudatoolkit=11.1 cudnn=8.2.1
pip install -r requirements.txt
cd d4rl
pip install -e .
cd ..
```
2. Install Jax and Jaxlib

- jax 0.4.9
- jaxlib 0.4.9(https://storage.googleapis.com/jax-releases/jax_cuda_releases.html)


## Run the code


### Run Training Reward Model

```python
CUDA_VISIBLE_DEVICES=0 python -m JaxPref.main --use_human_label True --comment {experiment_name} --transformer.embd_dim 256 --transformer.n_layer 3 --transformer.n_head 4 --env {D4RL env name} --logging.output_dir './logs/pref_reward' --batch_size 256 --num_query {number of query} --query_len 100 --n_epochs 10000 --skip_flag 0 --seed {seed} --model_type PrefMMT
```

### Run IQL with learned Reward Model

```python
# Preference Transfomer (PT)
CUDA_VISIBLE_DEVICES=0 python train_offline.py --seq_len {sequence length in reward prediction} --comment {experiment_name} --eval_interval {5000: mujoco / 100000: antmaze / 5000: metaworld} --env_name {d4rl env name} --config {configs/(mujoco|antmaze|metaworld)_config.py} --eval_episodes {100 for ant , 10 o.w.} --use_reward_model True --model_type PrefMMT --ckpt_dir {reward_model_path} --seed {seed}
```
(The code was tested in Ubuntu 20.04 with Python 3.8.)


## Acknowledgments

Our code is based on the implementation of[PT](https://github.com/csmile-1006/PreferenceTransformer), [Flaxmodels](https://github.com/matthias-wright/flaxmodels) and [IQL](https://github.com/ikostrikov/implicit_q_learning). 