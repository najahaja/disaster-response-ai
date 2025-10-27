import gymnasium as gym
from abc import ABC, abstractmethod
import numpy as np

class ActionSpace(ABC):
    """Abstract base class for action spaces"""
    
    @abstractmethod
    def get_action_space(self, agent_type):
        """Get action space for specific agent type"""
        pass
    
    @abstractmethod
    def decode_action(self, action, agent):
        """Decode numerical action to meaningful command"""
        pass

class DiscreteActionSpace(ActionSpace):
    """Discrete action space for grid movement"""
    
    def __init__(self):
        # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY, 5=REST
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'REST']
        self.action_space = gym.spaces.Discrete(len(self.actions))
    
    def get_action_space(self, agent_type):
        """Return action space (same for all agent types in basic version)"""
        return self.action_space
    
    def decode_action(self, action, agent):
        """Convert numerical action to movement command"""
        if 0 <= action < len(self.actions):
            return self.actions[action]
        return 'STAY'  # Default action
    
    def get_available_actions(self, agent, grid):
        """Get list of available actions considering obstacles"""
        available_actions = []
        
        for i, action in enumerate(self.actions):
            if action == 'STAY' or action == 'REST':
                available_actions.append(i)
            else:
                # Check if movement action is valid
                new_pos = agent._calculate_new_position(action)
                if agent._is_valid_move(new_pos, grid):
                    available_actions.append(i)
        
        return available_actions
    
    def __str__(self):
        return f"DiscreteActionSpace(actions={self.actions})"