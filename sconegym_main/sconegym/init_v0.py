import os
from gym.envs.registration import register
from pathlib import Path

curr_dir = Path(__file__).resolve()
register(id="H1922-v1",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': str(curr_dir.parent.parent.parent / "model" / "scone_model" / "H1922.scone"),
             'obs_type': '3D',
             'left_leg_idxs': [6, 7, 8, 9, 10],
             'right_leg_idxs': [11, 12, 13, 14, 15],
             'clip_actions': False,
             'run': False,
             'target_vel': 1.2,
             'leg_switch': False,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": 0,
                "joint_limit_coeff": 0,
                "smooth_coeff": 0,
                "nmuscle_coeff": 0,
                "self_contact_coeff": 0.0,
             }
         },
        )
register(id="H2190-v1",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': str(curr_dir.parent.parent.parent / "model" / "scone_model" / "H2190.scone"),
             'obs_type': '3D',
             'left_leg_idxs': [6, 7, 8, 9, 10, 11],
             'right_leg_idxs': [12, 13, 14, 15, 16, 17],
             'clip_actions': False,
             'run': False,
             'target_vel': 1.2,
             'leg_switch': False,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": 0,
                "joint_limit_coeff": 0,
                "smooth_coeff": 0,
                "nmuscle_coeff": 0,
                "self_contact_coeff": 0.0,
             }
         },
        )
# 2D Test environment
register(id="sconewalk_h0918-v1",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': str(curr_dir / 'data-v1' / 'H0918.scone'),
             'obs_type': '2D',
             'left_leg_idxs': [3, 4, 5],
             'right_leg_idxs': [6, 7, 8],
             'clip_actions': True,
             'run': False,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": -0.07281,
                "joint_limit_coeff": -0.1307,
                "smooth_coeff": -0.097,
                "nmuscle_coeff": -1.57929,
                "self_contact_coeff": 0.0,
             }
         }
        )