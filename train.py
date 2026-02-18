import os
import gym
import torch.nn as nn
import yaml
from pathlib import Path
from multiprocessing import freeze_support
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable
import datetime
import env


BASE = Path(__file__).resolve().parent
RESULT_DIR = BASE / "results"
PARAMS_DIR = BASE / "params"

save_name = datetime.datetime.now().strftime("%H-%M-%S")
n_envs = 30

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def make_env():
    def _init():
        env = gym.make('H0918-v0')
        env.render_mode = getattr(env, "render_mode", "rgb_array")
        return env
    return _init


def get_env():
    train_env = make_vec_env(
        make_env(), 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv, 
        vec_env_kwargs=dict(start_method='fork'))
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    return train_env


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
        use_sde=False,
        sde_sample_freq=-1)
    return model


if __name__ == '__main__':
    freeze_support()

    config_path = PARAMS_DIR / "cfg.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_env = get_env()
    model = get_model(config)
    
    model.learn(
        total_timesteps=config['total_timesteps'],
        log_interval=500,
        progress_bar=True)

    model.save(os.path.join(RESULT_DIR, save_name, "final_model"))
    train_env.save(os.path.join(RESULT_DIR, save_name, "final_model_vecnormalize.pkl"))
    train_env.close()