import pygame
import numpy as np
import yaml
import sys
import os

# Add parent directory to path to import from agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.base_env import BaseEnvironment
from environments.utils.visualization import VisualizationUtils
from environments.utils.disaster_generator import DisasterGenerator

class SimpleGridEnv(BaseEnvironment):
    """
    Simple grid-based environment for disaster response simulation
    """
    
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        super().__init__(self.config)
        
        # Environment parameters
        self.grid_size = self.config['environment']['grid_size']
        self.cell_size = self.config['environment']['cell_size']
        self.cell_types = self.config['environment']['cell_types']
        self.colors = self.config['visualization']['colors']
        
        # Environment state
        self.grid = None
        self.civilians = []
        self.collapsed_buildings = []
        self.blocked_roads = []
        
        # PyGame setup
        self.screen = None
        self.font = None
        self.initialize_pygame()
        
        # Initialize environment
        self.reset()
    
    def initialize_pygame(self):
        """Initialize PyGame for visualization"""
        pygame.init()
        width = self.grid_size * self.cell_size + 200  # Extra space for info panel
        height = self.grid_size * self.cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AI Disaster Response Simulation - Simple Grid")
        self.font = pygame.font.Font(None, 24)
    
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
    
    def reset(self):
        """Reset the environment to initial state"""
        self.step_count = 0
        self.disaster_triggered = False
        self.agents = {}
        self.civilians = []
        self.collapsed_buildings = []
        self.blocked_roads = []
        
        self.initialize_grid()
        return self.get_observation()
    
    def step(self, actions):
        """
        Execute one simulation step
        Args:
            actions: Dictionary of {agent_id: action}
        Returns:
            observation, rewards, done, info
        """
        self.step_count += 1
        
        # Execute actions for each agent
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                success = agent.move(action, self.grid)
                # You can use success to give rewards/penalties later
        
        # Check for civilian rescues (simple version - if agent is on civilian)
        self._check_civilian_rescues()
        
        # Calculate rewards
        rewards = self.calculate_rewards()
        
        # Check if episode is done
        done = self.is_done() or self._all_civilians_rescued()
        
        info = {
            'step': self.step_count,
            'civilians_rescued': sum(1 for c in self.civilians if c['rescued']),
            'total_civilians': len(self.civilians)
        }
        
        return self.get_observation(), rewards, done, info
    
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
        """Get current observation of the environment"""
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
        """Render the current state using PyGame"""
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
        
        pygame.display.flip()
    
    def trigger_disaster(self):
        """Trigger disaster scenario"""
        if self.disaster_triggered:
            return
            
        self.disaster_triggered = True
        self.grid, self.civilians, self.collapsed_buildings, self.blocked_roads = \
            DisasterGenerator.generate_disaster(self.grid, self.config)
    
    def add_agent(self, agent):
        """Add an agent to the environment"""
        super().add_agent(agent)
        print(f"✅ Added {agent.agent_id} at position {agent.position}")
    
    def close(self):
        """Close the environment"""
        pygame.quit()
    
    def __del__(self):
        """Destructor to ensure PyGame closes properly"""
        self.close()