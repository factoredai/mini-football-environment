from abc import ABC, abstractmethod
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment
import numpy as np
from gym import spaces

class UnityEnvironmentAbstract(ABC):
    @abstractmethod
    def step(action):
        pass

    @abstractmethod
    def reset():
        pass

    @abstractmethod
    def close():
        pass

class MiniFootballEnv(UnityEnvironmentAbstract):
    def __init__(self, path='./mini_football_windows/Mini Football Environment.exe', seed=0):

        # Initialize environment
        self.channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(path, seed=seed, side_channels=[self.channel])
        self.state = None

        self.env.step()
        self.behavior_name = list(self.env.behavior_specs)[0]
        self.reset()

        # Set Atributtes
        self.action_space = spaces.Tuple((spaces.Box(low=-1, high=1, shape=(2,)), spaces.Discrete(2)))
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(30,))
        self.reward_range = (-0.1, float("inf"))
        self.spec = None
        self.metadata = None
        self.np_random = seed

    def step(self, action):

        terminated = False
        truncated = False
        info = None
        step_reward = 0

        if len(action)==3:
            transformed_action = ActionTuple(np.array(action[:2]).reshape(1,2).astype(np.float32), np.array(action[-1]).reshape(1,1).astype(np.float32))
        elif len(action)==2 and len(action[0])==2:
            transformed_action = ActionTuple(np.array(action[0]).reshape(1,2).astype(np.float32), np.array(action[-1]).reshape(1,1).astype(np.float32))
        else:
            raise Exception("Invalid Action")

        self.env.set_actions(behavior_name=self.behavior_name, action=transformed_action)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        step_reward += decision_steps.reward[0]

        for agent_id in terminal_steps:
            step_reward += terminal_steps.reward[0]
            terminated = True

        if not terminated:
            observation = decision_steps.obs[0][0]
            
        else:
            observation = []
        
        self.state = observation

        return observation, step_reward, terminated, truncated, info

    def reset(self):
        self.env.reset()
        self.env.step()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        self.state = decision_steps.obs[0][0]

    def close(self):
        self.env.close()

    def set_channel_params(self, width, height, quality_level, time_scale, target_frame_rate, capture_frame_rate):
        self.channel.set_configuration_parameters(
            width= width,
            height= height,
            quality_level= quality_level,
            time_scale= time_scale,
            target_frame_rate= target_frame_rate,
            capture_frame_rate= capture_frame_rate,
            )
        
    def render():
        print("Render is automatically done, unless using headless build.")
