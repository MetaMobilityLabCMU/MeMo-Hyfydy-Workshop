import gym
import numpy as np
from gym.spaces import Box
from typing import Optional
from sconegym_main.sconetools import sconepy

np.random.seed(None)

class SconeEnv(gym.Env):

    def __init__(self, **kwargs):
        super().__init__()

        # Initialize sconepy model
        sconepy.set_log_level(3)
        self.model = sconepy.load_model(kwargs['model_file'])

        # Initiallize variables
        self.steps = 0
        self.time = 0
        self.episode = 0
        self.total_reward = 0
        self.step_size = 0.1 # 10 Hz
        self.max_step_size = int(3 / self.step_size) # default 3 seconds
        self.store_next = False
        self._bodies = {body.name(): body for body in self.model.bodies()}
        
        # Set action/observation space dimensions
        act_dim = len(self.model.actuators())
        self.action_space = Box(low=-np.ones(act_dim, dtype=np.float32),
                                high=np.ones(act_dim, dtype=np.float32),
                                dtype=np.float32)
        dummy_obs = self._get_obs()
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=dummy_obs.shape, dtype=np.float32)
        
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Reset environment
        self.model.reset()
        self.model.set_store_data(self.store_next)
        self.total_reward = 0.0
        self.steps = 0
        self.time = 0.0

        # Initialize muscle activations to small value
        muscle_activations = np.ones((len(self.model.muscles()),)) * 0.01
        self.model.init_muscle_activations(muscle_activations)

        # Equilibrate forces
        self.model.adjust_state_for_load(0.1)

        # Get initial observation
        obs = self._get_obs()
        info = {}
        return obs, info
    

    def step(self, action):
        self.steps += 1

        # Set muscle excitation from action
        action = np.clip(action, -1.0, 1.0)
        action = 0.5 * (action + 1.0) # map [-1, 1] to [0, 1]
        self.model.set_actuator_inputs(action)
        
        # Step simulation env
        self.time += self.step_size
        self.model.advance_simulation_to(self.time)
        
        # Get observation and reward
        obs = self._get_obs()
        reward = self._get_reward()
        
        # Check episode end
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        self.done = terminated or truncated

        # Store results
        self.total_reward += reward
        info = {}
        if self.done:
            self.episode += 1
            if self.store_next:
                self.model.write_results(sconepy.replace_string_tags("DATE_TIME."),
                                         f"{self.total_reward:.3f}")
                self.store_next = False
        
        return obs, reward, terminated, truncated, info
    

    def _get_obs(self) -> np.ndarray:        
        # Muscles
        m_fibl = self.model.muscle_fiber_length_array()
        m_fibv = self.model.muscle_fiber_velocity_array()
        m_force = self.model.muscle_force_array()
        m_exc = self.model.muscle_excitation_array()
        acts = self.model.muscle_activation_array()

        # Get model kinematics
        dof_pos = self.model.dof_position_array().copy()
        dof_vel = self.model.dof_velocity_array().copy()
        
        # Concat all observations
        obs = np.concatenate([
            m_fibl, m_fibv, m_force, m_exc, acts,
            dof_pos, dof_vel
        ], dtype=np.float32)

        return obs


    def _get_reward(self) -> float:
        reward = self._bodies['torso'].com_vel().y
        return reward
    

    def _is_terminated(self) -> bool:
        return False


    def _is_truncated(self) -> bool:
        if self.steps >= self.max_step_size:
            return True
        return False
    

    def store_next_episode(self):
        self.store_next = True
        self.reset()
    