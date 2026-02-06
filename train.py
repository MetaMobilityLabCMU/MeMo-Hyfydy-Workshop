import os
import gym
import torch.nn as nn
import yaml
from pathlib import Path
from multiprocessing import freeze_support
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor, DummyVecEnv, sync_envs_normalization
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable
import datetime
import env


BASE = Path(__file__).resolve().parent
RESULT_DIR = BASE / "results"
PARAMS_DIR = BASE / "params"

save_name = datetime.datetime.now().strftime("%m-%d_%H-%M")

n_envs = 10
save_freq = 10000
eval_freq = 10000


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def make_env():
    def _init():
        env = gym.make('scone_env-v0')
        env.render_mode = getattr(env, "render_mode", "rgb_array")
        return env
    return _init


def get_envs():
    train_env = make_vec_env(
        make_env(), 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv, 
        vec_env_kwargs=dict(start_method='fork'))
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
    
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.training = False
    eval_env.norm_reward = False

    return train_env, eval_env


def get_callback():
    checkpoint_callback = CheckpointCallback(
        save_freq=int(save_freq/n_envs),
        save_path=os.path.join(RESULT_DIR / save_name, "checkpoints"),
        name_prefix="sac_model",
        save_vecnormalize=True,
        save_replay_buffer=False)
    return checkpoint_callback


def get_model(config):
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=config['actor_net'], qf=config['critic_net']))
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=linear_schedule(config['lr_schedule']),
        buffer_size=config['buffer_size'],
        learning_starts=config['learning_starts'],
        batch_size=config['batch_size'],
        tau=config['tau'],
        gamma=config['gamma'],
        train_freq=config['train_freq'],
        gradient_steps=config['gradient_steps'],
        target_update_interval=config['target_update_interval'],
        ent_coef="auto",
        target_entropy="auto",
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(RESULT_DIR / save_name, "tensorboard_log"),
        verbose=1,
        device="auto",
        use_sde=True,
        sde_sample_freq=-1)
    return model


if __name__ == '__main__':
    freeze_support()

    config_path = PARAMS_DIR / "cfg.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_env, eval_env = get_envs()
    callback = get_callback()
    model = get_model(config)
    
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callback,
        log_interval=500,
        progress_bar=True)

    model.save(os.path.join(RESULT_DIR, "final_model"))
    train_env.save(os.path.join(RESULT_DIR, "final_model_vecnormalize.pkl"))
    train_env.close()
    eval_env.close()