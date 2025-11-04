import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import yaml
import random
import sys
import os
from agents.drone_agent import DroneAgent
from agents.ambulance_agent import AmbulanceAgent
from agents.rescue_team_agent import RescueTeamAgent
from agents.base_agent import BaseAgent

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.utils.visualization import VisualizationUtils
from environments.utils.disaster_generator import DisasterGenerator

class SimpleGridEnv(gym.Env):
    """
    Simple grid-based environment for disaster response simulation
    """
    
    def __init__(self, grid_size=None, config_path="config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Gymnasium required attributes
        self.metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(6)  # 6 actions: UP, DOWN, LEFT, RIGHT, STAY, REST

        # --- START OF FIX ---

        # 1. Determine grid size: Prioritize the 'grid_size' argument if passed,
        #    otherwise use the value from the config file.
        if grid_size is not None:
            self.grid_size = grid_size
        else:
            self.grid_size = self.config['environment']['grid_size']

        # 2. Define observation_space *using the grid_size variable*
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(self.grid_size, self.grid_size, 3),  # <-- WAS HARD-CODED to (15, 15, 3)
            dtype=np.uint8
        )
        
        # --- END OF FIX ---
        
        # Environment parameters
        # self.grid_size = self.config['environment']['grid_size'] # <-- This line is now handled above
        self.cell_size = self.config['environment']['cell_size']
        self.cell_types = self.config['environment']['cell_types']
        self.colors = self.config['visualization']['colors']
        self.max_steps = self.config['environment']['max_steps']
        
        # Environment state
        self.grid = None
        self.agents = {}
        self.civilians = []
        self.collapsed_buildings = []
        self.blocked_roads = []
        self.disaster_triggered = False
        self.step_count = 0
        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size
        # PyGame setup
        self.screen = pygame.Surface((self.width, self.height))
        self.font = None
        self.display_available = False
        
        self.initialize_pygame()
        self.reset()
    def _check_termination_conditions(self):
        """
        This is the method definition.
        Place your termination logic here and return the result.
        """
        # This logic will run when the method is called
        all_are_rescued = all(c['rescued'] for c in self.civilians)
        
        # Return the final True/False value
        return all_are_rescued
    def initialize_pygame(self):
        """Initialize PyGame for visualization - FORCED HEADLESS for Streamlit"""
        pygame.init()
        
        # We are using this with Streamlit, so we force headless mode
        # by not calling pygame.display.set_mode()
        width = self.grid_size * self.cell_size + 200
        height = self.grid_size * self.cell_size
        
        # This creates a "virtual" screen (a Surface) that doesn't open a window
        self.screen = pygame.Surface((width, height)) 
        self.font = pygame.font.Font(None, 24)
        self.display_available = False # Set to False so pygame.display.flip() is never called
        
        # We can still "print" that the GUI is (virtually) available for our app
        print("✅ PyGame initialized in headless mode for Streamlit.")
    
    def initialize_grid(self):
        """Initialize the grid with buildings, roads, and hospitals"""
        self.grid = np.full((self.grid_size, self.grid_size), 
                           self.cell_types['BUILDING'])
        
        # Create main roads (every 3rd row and column)
        for i in range(0, self.grid_size, 3):
            if i < self.grid_size:
                self.grid[i, :] = self.cell_types['ROAD']
                self.grid[:, i] = self.cell_types['ROAD']
        
        # Add hospitals at strategic locations
        hospital_positions = [
            [1, 1], [1, self.grid_size-2], 
            [self.grid_size-2, 1], [self.grid_size-2, self.grid_size-2]
        ]
        for pos in hospital_positions:
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                self.grid[pos[0], pos[1]] = self.cell_types['HOSPITAL']
    
    def reset(self,*, seed=None, options=None):
        """
        Reset the environment - gymnasium compatible
        """
        
        super().reset(seed=seed)
        
        self.step_count = 0
        self.disaster_triggered = False
        self.agents = {}
        self.civilians = []
        self.collapsed_buildings = []
        self.blocked_roads = []
        
        self.initialize_grid()
        try:
            # You can place them at hospitals or other known safe spots
            # Inside your reset() method
            self.add_agent(DroneAgent(agent_id="drone_0", position=[1, 1], config=self.config))
            self.add_agent(AmbulanceAgent(agent_id="ambulance_0", position=[1, self.grid_size-2], config=self.config))
            self.add_agent(RescueTeamAgent(agent_id="rescue_0", position=[self.grid_size-2, 1], config=self.config))
            # self.add_agent(BaseAgent(agent_id="base_0", position=[self.grid_size-2, 1], config=self.config))
        
        except NameError as e:
            print(f"ERROR: You forgot to import an agent class. {e}")
            # ... (error message) ...

        # 3. Trigger disaster (this creates civilians)
        # self.trigger_disaster()
        # Return observation and info
        observation = self._get_gym_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Execute one simulation step - modified for single-agent gym interface
        Compatible with Gymnasium (returns 5 values)
        """
        self.step_count += 1

        # Handle multi-agent or single-agent action input
        if isinstance(action, dict):
            # Multi-agent mode
            for agent_id, action in action.items():
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    if isinstance(action, int):
                        action_str = self._decode_action(action)
                    else:
                        action_str = action
                    agent.move(action_str, self.grid)
        else:
            # Single-agent mode (for gym compatibility)
            if self.agents:
                agent_id = list(self.agents.keys())[0]
                agent = self.agents[agent_id]
                action_str = self._decode_action(action)
                agent.move(action_str, self.grid)

        # Check for civilian rescues
        self._check_civilian_rescues()

        # Calculate rewards
        rewards = self.calculate_rewards()
        total_reward = sum(rewards.values()) if isinstance(rewards, dict) else rewards

        # Determine episode end conditions
        terminated = self._all_civilians_rescued()
        truncated = self.step_count >= self.max_steps

        # Compute observation
        obs = self._get_gym_observation()

        info = {
            'step': self.step_count,
            'civilians_rescued': sum(1 for c in self.civilians if c['rescued']),
            'total_civilians': len(self.civilians),
            'agents_count': len(self.agents)
        }

        return obs, total_reward, terminated, truncated, info

    
    def _decode_action(self, action):
        """Decode numeric action to string"""
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'REST']
        return actions[action] if 0 <= action < len(actions) else 'STAY'
    
    def _get_gym_observation(self):
        """Get observation in gym format"""
        # Create RGB observation
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
    
    def _check_civilian_rescues(self):
        """Check if any agents have rescued civilians"""
        for civilian in self.civilians:
            if not civilian['rescued']:
                civ_pos = civilian['position']
                for agent in self.agents.values():
                    if (agent.position == civ_pos).all():
                        if agent.rescue_civilian():
                            civilian['rescued'] = True
                            break
    
    def _all_civilians_rescued(self):
        """Check if all civilians have been rescued"""
        return all(civ['rescued'] for civ in self.civilians) if self.civilians else False
    def get_observation(self):
        """Get current observation of the environment (original method)"""
        return {
            'grid': self.grid.copy(),
            'agents': {aid: agent.get_state() for aid, agent in self.agents.items()},
            'civilians': self.civilians.copy(),
            'step_count': self.step_count,
            'disaster_triggered': self.disaster_triggered
        }
    
    def calculate_rewards(self):
        """Calculate rewards for each agent"""
        rewards = {}
        for agent_id, agent in self.agents.items():
            # Simple reward structure
            reward = agent.civilians_rescued * 100  # +100 per civilian rescued
            reward -= 1  # -1 per step to encourage efficiency
            reward -= agent.steps_taken * 0.1  # Small penalty for moving too much
            rewards[agent_id] = reward
        return rewards
    
    def render(self):
        """Render the current state using PyGame and RETURN the surface"""
        
        self.screen.fill((0, 0, 0))  # Clear screen
        
        # Draw grid
        VisualizationUtils.draw_grid(self.screen, self.grid, self.colors, self.cell_size)
        
        # Draw civilians
        VisualizationUtils.draw_civilians(self.screen, self.civilians, self.cell_size)
        
        # Draw agents
        VisualizationUtils.draw_agents(self.screen, self.agents, self.cell_size, self.font)
        
        # Draw info panel
        VisualizationUtils.draw_info_panel(self.screen, self.step_count, 
                                        self.agents, self.civilians, self.font)
        
        # DO NOT call pygame.display.flip()
        
        # RETURN the surface for Streamlit to use
        return self.screen
    
    def trigger_disaster(self):
        """Trigger disaster scenario"""
        if self.disaster_triggered:
            return
            
        self.disaster_triggered = True
        self.grid, self.civilians, self.collapsed_buildings, self.blocked_roads = \
            DisasterGenerator.generate_disaster(self.grid, self.config)
    
    def add_agent(self, agent):
        """Add an agent to the environment"""
        self.agents[agent.agent_id] = agent
        print(f"✅ Added {agent.agent_id} at position {agent.position}")
    
    def remove_agent(self, agent_id):
        """Remove an agent from the environment"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            
    def get_agent_positions(self):
        """Get positions of all agents"""
        return {aid: agent.position for aid, agent in self.agents.items()}
        
    def is_done(self):
        """Check if episode is done"""
        return self.step_count >= self.max_steps
    
    def close(self):
        """Close the environment"""
        pygame.quit()
    
    def __del__(self):
        """Destructor to ensure PyGame closes properly"""
        self.close()
    
    def __str__(self):
        return f"{self.__class__.__name__}(agents={len(self.agents)}, steps={self.step_count})"