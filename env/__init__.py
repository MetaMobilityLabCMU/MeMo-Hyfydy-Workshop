from gym.envs.registration import register
from pathlib import Path

curr_dir = Path(__file__).resolve()

register(
    id='H0918-v0',
    entry_point="env.scone_env:SconeEnv",
    kwargs={
        'model_file': str(curr_dir.parent.parent / "models" / "H0918.scone"),
    }
)