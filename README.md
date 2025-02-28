# PrefMMT

This repository contains the source code for our paper: **"PrefMMT: Modeling Human Preferences in Preference-based Reinforcement Learning with Multimodal Transformers"**, submitted to the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025). For more information, visit our [project website](https://sites.google.com/view/prefmmt).

## Abstract

Preference-based Reinforcement Learning (PbRL) is a promising approach for aligning robot behaviors with human preferences, but its effectiveness relies on accurately modeling those preferences through reward models. Traditional methods often assume preferences are Markovian, neglecting the temporal dependencies within robot behavior trajectories that influence human evaluations. While recent approaches use sequence modeling to learn non-Markovian rewards, they overlook the multimodal nature of robot trajectories, consisting of both state and action elements. This oversight limits their ability to capture the intricate interplay between these modalities, which is critical in shaping human preferences.

In this work, we introduce **PrefMMT**, a multimodal transformer network designed to disentangle and model the state and action modalities separately. PrefMMT hierarchically leverages intra-modal temporal dependencies and inter-modal state-action interactions to capture complex preference patterns. Our experiments show that PrefMMT consistently outperforms state-of-the-art baselines on locomotion tasks from the D4RL benchmark and manipulation tasks from the Meta-World benchmark.

## Comparison: PrefMMT vs. Other Preference Modeling Methods

<div align="center">
  <img src="/figures/Comparison.jpg" alt="Comparison of PrefMMT with other methods" width="800"/>
</div>

The diagram above highlights the key distinctions between **PrefMMT** and other existing preference modeling methods. While traditional approaches often make Markovian assumptions and fail to capture the multimodal interactions between state and action, **PrefMMT** addresses this gap by leveraging a multimodal transformer architecture. This allows for a more accurate and dynamic understanding of human preferences by modeling both intra-modal and inter-modal dependencies.

## Architecture Overview

<div align="center">
  <img src="/figures/PrefMMTAR.PNG" alt="PrefMMT Architecture" width="400"/>
</div>

## Installation

### Requirements

1. Install dependencies:

    ```bash
    pip install --upgrade pip
    conda install -y -c conda-forge cudatoolkit=11.1 cudnn=8.2.1
    pip install -r requirements.txt
    cd d4rl
    pip install -e .
    cd ..
    ```

2. Install JAX and JAXlib:

    - `jax 0.4.9`
    - `jaxlib 0.4.9` (Install from [JAX CUDA releases](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html))


## Run the code


### Run Training Reward Model

```python
CUDA_VISIBLE_DEVICES=0 python -m JaxPref.main --use_human_label True --comment {experiment_name} --transformer.embd_dim 256 --transformer.n_layer 3 --transformer.n_head 4 --env {D4RL env name} --logging.output_dir './logs/pref_reward' --batch_size 256 --num_query {number of query} --query_len 100 --n_epochs 10000 --skip_flag 0 --seed {seed} --model_type PrefMMT
```

### Run IQL with learned Reward Model

```python
CUDA_VISIBLE_DEVICES=0 python train_offline.py --seq_len {sequence length in reward prediction} --comment {experiment_name} --eval_interval {5000: mujoco / 100000: antmaze / 5000: metaworld} --env_name {d4rl env name} --config {configs/(mujoco|antmaze|metaworld)_config.py} --eval_episodes {100 for ant , 10 o.w.} --use_reward_model True --model_type PrefMMT --ckpt_dir {reward_model_path} --seed {seed}
```
(The code was tested in Ubuntu 20.04 with Python 3.9.)


## Acknowledgments

Our code is based on the implementation of [PT](https://github.com/csmile-1006/PreferenceTransformer), [Flaxmodels](https://github.com/matthias-wright/flaxmodels) and [IQL](https://github.com/ikostrikov/implicit_q_learning). 
