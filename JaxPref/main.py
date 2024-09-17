import os
import pickle
from collections import defaultdict

import numpy as np

import transformers

import gym
import wrappers as wrappers

import absl.app
import absl.flags
from flax.training.early_stopping import EarlyStopping
from flaxmodels.flaxmodels.lstm.lstm import LSTMRewardModel
from flaxmodels.flaxmodels.gpt2.trajectory_gpt2 import TransRewardModel

from .sampler import TrajSampler
from .jax_utils import batch_to_jax
import JaxPref.reward_transform as r_tf
from .model import FullyConnectedQFunction
from viskit.logging import logger, setup_logger
from .MR import MR
from .replay_buffer import get_d4rl_dataset, index_batch
from .NMR import NMR
from .PrefMMT import PrefMMT
from .utils import Timer, define_flags_with_default, set_random_seed, get_user_flags, prefix_metrics, WandBLogger, save_pickle
from metaworld_utils import make_metaworld_env,get_dataset

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'

FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    model_type='MLP',
    max_traj_length=500,
    seed=42,
    data_seed=42,
    save_model=True,
    batch_size=64,
    early_stop=False,
    min_delta=1e-3,
    patience=10,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    reward_arch='256-256',
    orthogonal_init=False,
    activations='relu',
    activation_final='none',
    training=True,

    n_epochs=2000,
    eval_period=5,

    data_dir='./human_label',
    num_query=1000,
    query_len=25,
    skip_flag=0,
    balance=False,
    topk=10,
    window=2,
    use_human_label=False,
    feedback_random=False,
    feedback_uniform=False,
    enable_bootstrap=False,

    comment='',

    reward=MR.get_default_config(),
    transformer=PrefMMT.get_default_config(),
    lstm=NMR.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(_):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)

    save_dir = FLAGS.logging.output_dir + '/' + FLAGS.env
    save_dir += '/' + str(FLAGS.model_type) + '/'

    FLAGS.logging.group = f"{FLAGS.env}_{FLAGS.model_type}"
    assert FLAGS.comment, "You must leave your comment for logging experiment."
    FLAGS.logging.group += f"_{FLAGS.comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    save_dir += str(FLAGS.seed)

    setup_logger(
        variant=variant,
        seed=FLAGS.seed,
        base_log_dir=save_dir,
        include_exp_prefix_sub_dir=False
    )

    FLAGS.logging.output_dir = save_dir
    wb_logger = WandBLogger(FLAGS.logging, variant=variant)

    set_random_seed(FLAGS.seed)

    if 'ant' in FLAGS.env:
        gym_env = gym.make(FLAGS.env)
        gym_env = wrappers.EpisodeMonitor(gym_env)
        gym_env = wrappers.SinglePrecision(gym_env)
        gym_env.seed(FLAGS.seed)
        gym_env.action_space.seed(FLAGS.seed)
        gym_env.observation_space.seed(FLAGS.seed)
        dataset = r_tf.qlearning_ant_dataset(gym_env)
        label_type = 1
    if 'metaworld' in FLAGS.env:
        gym_env = make_metaworld_env(FLAGS.env,FLAGS.seed)
        gym_env = wrappers.EpisodeMonitor(gym_env)
        gym_env.action_space.seed(FLAGS.seed)
        gym_env.observation_space.seed(FLAGS.seed)
        gym_env._max_episode_steps = 500
        dataset = get_dataset(FLAGS.env,gym_env)
        label_type = 0
    else:
        gym_env = gym.make(FLAGS.env)
        eval_sampler = TrajSampler(gym_env.unwrapped, FLAGS.max_traj_length)
        dataset = get_d4rl_dataset(eval_sampler.env)
        label_type = 0

    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)
    # use fixed seed for collecting segments.
    set_random_seed(FLAGS.data_seed)

    print("load saved indices.")
    env = FLAGS.env

    base_path = os.path.join(FLAGS.data_dir, env)
    if os.path.exists(base_path):
        human_indices_2_file, human_indices_1_file, human_labels_file = sorted(os.listdir(base_path))
        with open(os.path.join(base_path, human_indices_1_file), "rb") as fp:   # Unpickling
            human_indices = pickle.load(fp)
        with open(os.path.join(base_path, human_indices_2_file), "rb") as fp:   # Unpickling
            human_indices_2 = pickle.load(fp)
        with open(os.path.join(base_path, human_labels_file), "rb") as fp:   # Unpickling
            human_labels = pickle.load(fp)

        pref_dataset = r_tf.load_queries_with_indices(
            gym_env, dataset, FLAGS.num_query, FLAGS.query_len,
            label_type=label_type, saved_indices=[human_indices, human_indices_2], saved_labels=human_labels,
            balance=FLAGS.balance, scripted_teacher=not FLAGS.use_human_label)

        true_eval = True if len(human_labels) > FLAGS.num_query else False
        pref_eval_dataset = r_tf.load_queries_with_indices(
            gym_env, dataset, int(FLAGS.num_query * 0.1), FLAGS.query_len,
            label_type=label_type, saved_indices=[human_indices, human_indices_2], saved_labels=human_labels,
            balance=FLAGS.balance, scripted_teacher=not FLAGS.use_human_label)
    else:
        pref_dataset = r_tf.get_queries_from_multi(
            gym_env, dataset, FLAGS.num_query, FLAGS.query_len,
            data_dir=base_path, label_type=label_type, balance=FLAGS.balance)

        human_indices_2_file, human_indices_1_file, script_labels_file = sorted(os.listdir(base_path))
        with open(os.path.join(base_path, human_indices_1_file), "rb") as fp:   # Unpickling
            human_indices = pickle.load(fp)
        with open(os.path.join(base_path, human_indices_2_file), "rb") as fp:   # Unpickling
            human_indices_2 = pickle.load(fp)
        with open(os.path.join(base_path, script_labels_file), "rb") as fp:   # Unpickling
            human_labels = pickle.load(fp)
        true_eval = True if len(human_labels) > FLAGS.num_query else False
        pref_eval_dataset = r_tf.load_queries_with_indices(
            gym_env, dataset, int(FLAGS.num_query * 0.1), FLAGS.query_len,
            label_type=label_type, saved_indices=[human_indices, human_indices_2], saved_labels=human_labels,
            balance=FLAGS.balance, topk=FLAGS.topk, scripted_teacher=True, window=FLAGS.window, 
            feedback_random=FLAGS.feedback_random, pref_attn_n_head=FLAGS.transformer.pref_attn_n_head, true_eval=true_eval)

    set_random_seed(FLAGS.seed)
    observation_dim = gym_env.observation_space.shape[0]
    action_dim = gym_env.action_space.shape[0]

    data_size = pref_dataset["observations"].shape[0]
    interval = int(data_size / FLAGS.batch_size) + 1

    eval_data_size = pref_eval_dataset["observations"].shape[0]
    eval_interval = int(eval_data_size / FLAGS.batch_size) + 1

    early_stop = EarlyStopping(min_delta=FLAGS.min_delta, patience=FLAGS.patience)

    if FLAGS.model_type == "MR":
        rf = FullyConnectedQFunction(observation_dim, action_dim, FLAGS.reward_arch, FLAGS.orthogonal_init, FLAGS.activations, FLAGS.activation_final)
        reward_model = MR(FLAGS.reward, rf)

    elif FLAGS.model_type == "PrefMMT":
        total_epochs = FLAGS.n_epochs
        config = transformers.GPT2Config(
            **FLAGS.transformer
        )
        config.warmup_steps = int(total_epochs * 0.1 * interval)
        config.total_steps = total_epochs * interval

        trans = TransRewardModel(config=config, observation_dim=observation_dim, action_dim=action_dim, activation=FLAGS.activations, activation_final=FLAGS.activation_final)
        reward_model = PrefMMT(config, trans)

    elif FLAGS.model_type == "NMR":
        total_epochs = FLAGS.n_epochs
        config = transformers.GPT2Config(
            **FLAGS.lstm
        )
        config.warmup_steps = int(total_epochs * 0.1 * interval)
        config.total_steps = total_epochs * interval

        lstm = LSTMRewardModel(config=config, observation_dim=observation_dim, action_dim=action_dim, activation=FLAGS.activations, activation_final=FLAGS.activation_final)
        reward_model = NMR(config, lstm)

    if FLAGS.model_type == "MR":
        train_loss = "reward/rf_loss"
    elif FLAGS.model_type == "NMR":
        train_loss = "reward/lstm_loss"
    elif FLAGS.model_type == "PrefMMT":
        train_loss = "reward/trans_loss"

    criteria_key = None
    for epoch in range(FLAGS.n_epochs + 1):
        metrics = defaultdict(list)
        metrics['epoch'] = epoch
        if epoch:
            # train phase
            shuffled_idx = np.random.permutation(pref_dataset["observations"].shape[0])
            for i in range(interval):
                start_pt = i * FLAGS.batch_size
                end_pt = min((i + 1) * FLAGS.batch_size, pref_dataset["observations"].shape[0])
                with Timer() as train_timer:
                    # train
                    batch = batch_to_jax(index_batch(pref_dataset, shuffled_idx[start_pt:end_pt]))
                    for key, val in prefix_metrics(reward_model.train(batch), 'reward').items():
                        metrics[key].append(val)
            metrics['train_time'] = train_timer()
        else:
            # for using early stopping with train loss.
            metrics[train_loss] = [float(FLAGS.query_len)]

        # eval phase
        if epoch % FLAGS.eval_period == 0:
            for j in range(eval_interval):
                eval_start_pt, eval_end_pt = j * FLAGS.batch_size, min((j + 1) * FLAGS.batch_size, pref_eval_dataset["observations"].shape[0])
                # batch_eval = batch_to_jax(index_batch(pref_eval_dataset, range(eval_start_pt, eval_end_pt)))
                batch_eval = batch_to_jax(index_batch(pref_eval_dataset, range(eval_start_pt, eval_end_pt)))
                for key, val in prefix_metrics(reward_model.evaluation(batch_eval), 'reward').items():
                    metrics[key].append(val)
            if not criteria_key:
                if "antmaze" in FLAGS.env and not "dense" in FLAGS.env and not true_eval:
                    # choose train loss as criteria.
                    criteria_key = train_loss
                else:
                    # choose eval loss as criteria.
                    criteria_key = key
            criteria = np.mean(metrics[criteria_key])
            has_improved, early_stop = early_stop.update(criteria)
            if early_stop.should_stop and FLAGS.early_stop:
                for key, val in metrics.items():
                    if isinstance(val, list):
                        metrics[key] = np.mean(val)
                logger.record_dict(metrics)
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
                wb_logger.log(metrics)
                print('Met early stopping criteria, breaking...')
                break
            elif epoch > 0 and has_improved:
                metrics["best_epoch"] = epoch
                metrics[f"{key}_best"] = criteria
                save_data = {"reward_model": reward_model, "variant": variant, "epoch": epoch}
                save_pickle(save_data, "best_model.pkl", save_dir)

        for key, val in metrics.items():
            if isinstance(val, list):
                metrics[key] = np.mean(val)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        wb_logger.log(metrics)

    if FLAGS.save_model:
        save_data = {'reward_model': reward_model, 'variant': variant, 'epoch': epoch}
        save_pickle(save_data, 'model.pkl', save_dir)


if __name__ == '__main__':
    absl.app.run(main)
