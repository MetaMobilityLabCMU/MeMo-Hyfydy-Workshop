import gym
import yaml
from argparse import ArgumentParser
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import env

BASE = Path(__file__).resolve().parent
RESULT_DIR = BASE / "results"
PARAMS_DIR = BASE / "params"

def make_env():
    def _init():
        env = gym.make('H0918-v0')
        env.render_mode = getattr(env, "render_mode", "rgb_array")
        return env
    return _init

def run(name):
    eval_env = DummyVecEnv([make_env()])

    result_dir = RESULT_DIR / name
    vecnormalize_path = result_dir / f"final_model_vecnormalize.pkl"

    eval_env = VecNormalize.load(vecnormalize_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model_path = result_dir / f"final_model.zip"
    model = SAC.load(model_path, device='auto')

    eval_env.envs[0].store_next_episode()
    obs = eval_env.reset()
    ep_reward = 0.0
    ep_steps = 0
    done = [False]

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        ep_reward += reward[0]
        ep_steps += 1

    eval_env.close()
    print(f"Rollout ended; steps = {ep_steps}; total reward = {ep_reward:.3f}")

if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument('name', type=str)
    args = argparse.parse_args()

    config_path = PARAMS_DIR / "cfg.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    run(args.name)