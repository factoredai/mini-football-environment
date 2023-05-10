from typing import Any
from abc import ABC, abstractmethod
import onnxruntime as rt
import numpy as np

class Agent(ABC):
    @abstractmethod
    def act(self, state: Any, epsilon: float = 0.0):
        """Choose an action given a state
        Args:
            state (Any): A representation of the state
            epsilon (float, optional): Probability of taking an exploratory action. Defaults to 0.0.
        """

    @abstractmethod
    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool):
        """Update the policy given the current experience
        Args:
            state (Any): A representation of the current state
            action (Any): Action taken at the current state
            reward (float): Reward obtained from taking action at the current state
            next_state (Any): State reached after acting on the current state
            done (bool): Wether the episode is over at the next state
        """

    @abstractmethod
    def save(self, path: str):
        """Store the current agent in memory
        Args:
            path (str): Where to store the agent
        """

    @abstractmethod
    def load(self, path: str):
        """Load an agent from memory
        Args:
            path (str): Where the agent is stored
        """


class MiniFootballAgent(Agent):
    def __init__(self):
        self.sess = None

    def load(self, path="trained_brains/FootballPlayer.onnx"):
        # Load Brain
        self.sess = rt.InferenceSession(path)

    def save():
        pass

    def act(self, state: Any, epsilon: float = 0.0):
        # Params
        input_name0 = self.sess.get_inputs()[0].name
        input_name1= self.sess.get_inputs()[1].name

        if len(state) == 0:
            return [0,0,0]
        
        state = np.concatenate((state[:15], state[3:6]-state[6:9], state[15:], state[15+3:15+6]-state[15+6:15+9]), axis=0)
        
        a = np.ones((2,2)).astype(np.float32)
        new_obs = state.astype(np.float32)

        pred = self.sess.run(['continuous_actions', 'discrete_actions'], {input_name0: np.array(new_obs).reshape(1,36), input_name1: a})
        return np.concatenate((pred[0][0], pred[1][0]), axis=0)

    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool):
        pass
