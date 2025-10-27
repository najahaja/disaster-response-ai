import gymnasium as gym
from abc import ABC, abstractmethod
import numpy as np

class ObservationSpace(ABC):
    """Abstract base class for observation spaces"""
    
    @abstractmethod
    def get_observation_space(self, agent_type, grid_size):
        """Get observation space for specific agent type"""
        pass
    
    @abstractmethod
    def encode_observation(self, agent, env_observation):
        """Environment state to numerical observation"""
        pass

class GlobalObservation(ObservationSpace):
    """Global observation - agent sees entire grid"""
    
    def __init__(self, grid_size, num_cell_types, num_agent_types):
        self.grid_size = grid_size
        self.num_cell_types = num_cell_types
        self.num_agent_types = num_agent_types
        
        # Observation: flattened grid + agent position + agent type
        self.obs_size = (grid_size * grid_size) + 2 + 1  # grid + pos + type
        self.observation_space = gym.spaces.Box(
            low=0, high=max(num_cell_types, num_agent_types, grid_size),
            shape=(self.obs_size,), dtype=np.float32
        )
    
    def get_observation_space(self, agent_type, grid_size):
        return self.observation_space
    
    def encode_observation(self, agent, env_observation):
        """Encode observation for a specific agent"""
        grid = env_observation['grid']
        agents = env_observation['agents']
        civilians = env_observation['civilians']
        
        # Flatten the grid
        grid_flat = grid.flatten()
        
        # Normalize agent position
        pos_x = agent.position[0] / self.grid_size
        pos_y = agent.position[1] / self.grid_size
        
        # Agent type encoding
        type_encoding = {
            'drone': 0.1,
            'ambulance': 0.2, 
            'rescue_team': 0.3
        }
        agent_type_enc = type_encoding.get(agent.agent_type, 0.0)
        
        # Combine all features
        observation = np.concatenate([
            grid_flat,
            np.array([pos_x, pos_y]),
            np.array([agent_type_enc])
        ])
        
        return observation

class PartialObservation(ObservationSpace):
    """Partial observation - agent sees only local area"""
    
    def __init__(self, grid_size, num_cell_types, num_agent_types, view_range=5):
        self.grid_size = grid_size
        self.view_range = view_range
        self.obs_size = (view_range * 2 + 1) ** 2 + 2 + 1  # local view + pos + type
        
        self.observation_space = gym.spaces.Box(
            low=0, high=max(num_cell_types, num_agent_types, grid_size),
            shape=(self.obs_size,), dtype=np.float32
        )
    
    def get_observation_space(self, agent_type, grid_size):
        return self.observation_space
    
    def encode_observation(self, agent, env_observation):
        """Encode local observation around agent"""
        grid = env_observation['grid']
        pos_x, pos_y = agent.position
        
        # Extract local view
        local_view = []
        for dy in range(-self.view_range, self.view_range + 1):
            for dx in range(-self.view_range, self.view_range + 1):
                view_x, view_y = pos_x + dx, pos_y + dy
                if 0 <= view_x < self.grid_size and 0 <= view_y < self.grid_size:
                    local_view.append(grid[view_y, view_x])
                else:
                    local_view.append(-1)  # Out of bounds
        
        # Normalize position and type
        pos_x_norm = pos_x / self.grid_size
        pos_y_norm = pos_y / self.grid_size
        
        type_encoding = {
            'drone': 0.1,
            'ambulance': 0.2,
            'rescue_team': 0.3
        }
        agent_type_enc = type_encoding.get(agent.agent_type, 0.0)
        
        # Combine features
        observation = np.concatenate([
            np.array(local_view),
            np.array([pos_x_norm, pos_y_norm]),
            np.array([agent_type_enc])
        ])
        
        return observation