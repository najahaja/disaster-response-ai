import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import yaml
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.simple_grid_env import SimpleGridEnv

class RealMapEnv(SimpleGridEnv):
    """
    Real map environment that falls back to SimpleGridEnv when map loading fails
    """
    
    def __init__(self, location_name=None, config_path="config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Gymnasium required attributes
        self.metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(20, 20, 3),  # Default grid size with RGB channels
            dtype=np.uint8
        )
        
        # Environment parameters
        self.grid_size = 20  # Default size
        self.cell_size = self.config['environment']['cell_size']
        self.cell_types = self.config['environment']['cell_types']
        self.colors = self.config['visualization']['colors']
        self.max_steps = self.config['environment']['max_steps']
        
        # Real map attributes
        self.location_name = location_name
        self.real_map_loaded = False
        
        # Environment state
        self.grid = None
        self.agents = {}
        self.civilians = []
        self.collapsed_buildings = []
        self.blocked_roads = []
        self.disaster_triggered = False
        self.step_count = 0
        
        # PyGame setup
        self.screen = None
        self.font = None
        self.display_available = False
        
        self.initialize_pygame()
        
        # Try to load real map, fallback to simple grid
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize environment with real map or fallback to simple grid"""
        if self.location_name:
            try:
                # Try to import and use map loader
                from environments.utils.map_loader import MapLoader
                
                map_loader = MapLoader()
                map_data = map_loader.load_map_with_fallback(self.location_name)
                
                if map_data and hasattr(map_data, 'grid'):
                    self.grid = map_data.grid
                    self.grid_size = self.grid.shape[0]
                    self.real_map_loaded = True
                    print(f"✅ Real map loaded for: {self.location_name}")
                    return
                else:
                    print(f"⚠️  Map data incomplete for {self.location_name}, using fallback")
            except ImportError as e:
                print(f"⚠️  Map loader not available: {e}")
            except Exception as e:
                print(f"⚠️  Failed to load real map: {e}")
        
        # Fallback to simple grid
        print("🔄 Using simple grid as fallback")
        self._initialize_simple_grid()
    
    def _initialize_simple_grid(self):
        """Initialize a simple grid as fallback"""
        self.grid_size = 20
        self.grid = np.full((self.grid_size, self.grid_size), 
                           self.cell_types['BUILDING'])
        
        # Create main roads
        for i in range(0, self.grid_size, 3):
            if i < self.grid_size:
                self.grid[i, :] = self.cell_types['ROAD']
                self.grid[:, i] = self.cell_types['ROAD']
        
        # Add hospitals
        hospital_positions = [
            [2, 2], [2, self.grid_size-3], 
            [self.grid_size-3, 2], [self.grid_size-3, self.grid_size-3]
        ]
        for pos in hospital_positions:
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                self.grid[pos[0], pos[1]] = self.cell_types['HOSPITAL']
        
        self.real_map_loaded = False
    
    def initialize_pygame(self):
        """Initialize PyGame for visualization"""
        try:
            pygame.init()
            width = self.grid_size * self.cell_size + 200
            height = self.grid_size * self.cell_size
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Real Map Disaster Response Simulation")
            self.font = pygame.font.Font(None, 24)
            self.display_available = True
            print("✅ GUI display available")
        except pygame.error as e:
            print(f"⚠️  No GUI display: {e}")
            self.screen = pygame.Surface((width, height))
            self.font = pygame.font.Font(None, 24)
            self.display_available = False
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.disaster_triggered = False
        self.agents = {}
        self.civilians = []
        self.collapsed_buildings = []
        self.blocked_roads = []
        
        # Reinitialize grid (in case we want to reload map)
        self._initialize_environment()
        
        observation = self._get_gym_observation()
        info = {'real_map_loaded': self.real_map_loaded}
        
        return observation, info
    
    def _get_gym_observation(self):
        """Get observation in gym format"""
        observation = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                cell_type = self.grid[y, x]
                color = self.colors.get(cell_type, (0, 0, 0))
                observation[y, x] = color
        
        # Add agents to observation
        for agent in self.agents.values():
            x, y = agent.position
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                observation[y, x] = agent.color
        
        return observation
    
    def __str__(self):
        map_type = "Real Map" if self.real_map_loaded else "Simple Grid"
        return f"RealMapEnv({map_type}, agents={len(self.agents)}, steps={self.step_count})"