import metaworld.envs.mujoco.env_dict as _env_dict
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import h5py
from tqdm import tqdm

def make_metaworld_env(env_id, seed):
    env_name = env_id.replace('metaworld_','')
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]
    
    env = env_cls()
    
    #env = metaworld.ML1(env_name,seed=seed)

    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)
    
    return TimeLimit(env, env.max_path_length)

def get_dataset(env_name,env):
    env_name = env_name.replace('metaworld_','').replace('-v2','')
    data = h5py.File(f'./data/metaworld/{env_name}.hdf5','r')
    obs = data['observations']
    action = data['actions']
    reward = data['rewards']
    done = data['terminals']
    N = obs.shape[0]
    next_obs_ = [] 
    done_ = []
    done_bef_ = []
    episode_step = 0
    for i in tqdm(range(N - 1),desc="getting dataset:"):
        new_obs = obs[i + 1].astype(np.float32)
        done_bool = bool(data["terminals"][i]) or episode_step == env._max_episode_steps - 1
        
        final_timestep = episode_step == env._max_episode_steps - 1
        next_final_timestep = episode_step == env._max_episode_steps - 2

        done_bef = bool(next_final_timestep)

        next_obs_.append(new_obs)
        done_.append(done_bool)
        if final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        done_bef_.append(done_bef)
        episode_step += 1
    print(' Get Offline Dataset ! ')
    return {
        "observations": obs,
        "actions": action,
        "rewards" : reward,
        "next_observations": np.array(next_obs_),
        "terminals": np.array(done_),
        "dones_bef": np.array(done_bef_),
    }