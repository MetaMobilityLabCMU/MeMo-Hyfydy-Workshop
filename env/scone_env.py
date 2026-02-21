import gym
import numpy as np
from gym.spaces import Box
from typing import Optional
from sconegym_main.sconetools import sconepy

np.random.seed(None)

class SconeEnv(gym.Env):

    def __init__(self, **kwargs):
        """
        Initialize environment.
        You can add any class variables here.
        """

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
        self.max_step_size = int(0.8 / self.step_size) # default 0.8 seconds
        self.store_next = False
        
        # Set action/observation space dimensions
        act_dim = len(self.model.actuators())
        self.action_space = Box(low=-np.ones(act_dim, dtype=np.float32),
                                high=np.ones(act_dim, dtype=np.float32),
                                dtype=np.float32)
        dummy_obs = self._get_obs()
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=dummy_obs.shape, dtype=np.float32)
        
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets environment.
        Returns initial observation.
        """

        # Initialize environment and variables
        self.model.reset()
        self.model.set_store_data(self.store_next)
        self.total_reward = 0.0
        self.steps = 0
        self.time = 0.0

        # Initialize muscle activations to small value
        muscle_activations = np.ones((len(self.model.muscles()),)) * 0.01
        self.model.init_muscle_activations(muscle_activations)

        # Adjust model vertical displacement to 10% of body weight
        self.model.adjust_state_for_load(0.1)

        # Get initial observation
        obs = self._get_obs()
        info = {}
        return obs, info
    

    def step(self, action):
        """
        Takes action inside the Hyfydy simulation and returns:
        (obs, reward, terminated, truncated, info).
        """

        self.steps += 1
        self.time += self.step_size

        # Set muscle excitation from action
        action = np.clip(action, -1.0, 1.0)
        action = 0.5 * (action + 1.0) # map [-1, 1] to [0, 1]
        self.model.set_actuator_inputs(action)
        
        # Step simulation env
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
        """
        Returns observation vector.
        This vector goes into the neural networks.
        """

        # TODO: Give useful observations to the model
        NotImplemented
        
        return np.array([0])


    def _get_reward(self) -> float:
        """ Returns scaler reward of the step. """

        # TODO: Design reward function!
        NotImplemented

        return 0.0
    

    def _is_terminated(self) -> bool:
        """ Checks episode termination. """

        # TODO: (optional) Set early termination condition

        return False


    def _is_truncated(self) -> bool:
        """ Checks if the episode reached max steps. """
        if self.steps >= self.max_step_size:
            return True
        return False
    

    def store_next_episode(self):
        """ Set save conditions of the next episode. """
        self.store_next = True
        self.reset()