import numpy as np
import pygame
import yaml
import osmnx as ox
import networkx as nx
from typing import Dict, List, Tuple
import random

from environments.simple_grid_env import SimpleGridEnv
from environments.utils.map_loader import MapLoader
from environments.utils.dynamic_events import DynamicEventManager

class RealMapEnv(SimpleGridEnv):
    """
    Real-world map environment using OpenStreetMap data
    """
    
    def __init__(self, location_name="Lahore, Pakistan", config_path="config.yaml"):
        # Load configuration first
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize map loader
        self.map_loader = MapLoader()
        self.location_name = location_name
        
        # Load real map
        self.graph = None
        self.node_positions = None
        self.edge_data = None
        self.grid = None
        
        # Initialize dynamic events
        self.event_manager = DynamicEventManager(self.config)
        
        # Call parent constructor after map is loaded
        super().__init__(config_path)
        
        # Override grid with real map
        self.load_real_map()
    
    def load_real_map(self):
        """Load real map data from OpenStreetMap"""
        print(f"🗺️  Loading map for {self.location_name}...")
        
        try:
            # Get street network
            self.graph = ox.graph_from_place(
                self.location_name, 
                network_type='drive', 
                simplify=True
            )
            
            # Convert to grid representation
            self.grid, self.node_positions, self.edge_data = \
                self.map_loader.graph_to_grid(self.graph, grid_size=20)
            
            print(f"✅ Map loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            print(f"✅ Grid size: {self.grid.shape}")
            
        except Exception as e:
            print(f"❌ Failed to load real map: {e}")
            print("🔄 Falling back to generated grid...")
            self.initialize_grid()
    
    def initialize_grid(self):
        """Override to use real map data"""
        if self.grid is not None:
            # Use the real map grid
            self.grid_size = self.grid.shape[0]
        else:
            # Fallback to generated grid
            super().initialize_grid()
    
    def add_agent(self, agent):
        """Add agent to a valid position on the real map"""
        if self.node_positions:
            # Place agent at a random node position
            valid_nodes = list(self.node_positions.keys())
            if valid_nodes:
                node_id = random.choice(valid_nodes)
                position = self.node_positions[node_id]
                agent.position = np.array(position, dtype=int)
                super().add_agent(agent)
                print(f"✅ Added {agent.agent_id} at node {node_id} (position {position})")
            else:
                super().add_agent(agent)
        else:
            super().add_agent(agent)
    
    def step(self, actions):
        """Step with dynamic events"""
        # Check for dynamic events
        events = self.event_manager.check_events(self.step_count, self)
        
        # Execute normal step
        observation, rewards, done, info = super().step(actions)
        
        # Add event information
        info['events'] = events
        
        # Apply event effects
        for event in events:
            self._apply_event(event)
        
        return observation, rewards, done, info
    
    def _apply_event(self, event):
        """Apply dynamic event effects"""
        if event['type'] == 'aftershock':
            # Additional building collapses
            self._trigger_aftershock(event['intensity'])
        elif event['type'] == 'resource_depletion':
            # Reduce agent capacities
            self._deplete_resources(event['agent_type'])
        elif event['type'] == 'road_closure':
            # Block additional roads
            self._close_roads(event['locations'])
    
    def _trigger_aftershock(self, intensity):
        """Trigger aftershock - collapse more buildings"""
        building_positions = np.argwhere(self.grid == self.cell_types['BUILDING'])
        num_collapses = min(int(intensity * 5), len(building_positions))
        
        for pos in random.sample(list(building_positions), num_collapses):
            self.grid[tuple(pos)] = self.cell_types['COLLAPSED']
            # Chance to spawn additional civilians
            if random.random() < 0.2:
                self.add_civilian(pos.tolist())
        
        print(f"💥 Aftershock! {num_collapses} additional buildings collapsed")
    
    def _deplete_resources(self, agent_type):
        """Deplete resources for specific agent type"""
        for agent in self.agents.values():
            if agent.agent_type == agent_type:
                # Reduce capacity temporarily
                original_capacity = getattr(agent, '_original_capacity', agent.capacity)
                agent._original_capacity = original_capacity
                agent.capacity = max(1, original_capacity - 1)
                print(f"⚡ Resource depletion: {agent.agent_id} capacity reduced to {agent.capacity}")
    
    def _close_roads(self, locations):
        """Close additional roads"""
        for pos in locations:
            if (0 <= pos[0] < self.grid_size and 
                0 <= pos[1] < self.grid_size and
                self.grid[pos[1], pos[0]] == self.cell_types['ROAD']):
                self.grid[pos[1], pos[0]] = self.cell_types['BLOCKED']
        
        print(f"🚧 Road closures: {len(locations)} roads blocked")
    
    def get_map_info(self):
        """Get information about the real map"""
        if self.graph:
            return {
                'location': self.location_name,
                'nodes': len(self.graph.nodes),
                'edges': len(self.graph.edges),
                'grid_size': self.grid_size,
                'is_real_map': True
            }
        else:
            return {
                'location': 'Generated',
                'nodes': 0,
                'edges': 0,
                'grid_size': self.grid_size,
                'is_real_map': False
            }
    
    def render(self):
        """Render with real map features"""
        if hasattr(self, 'display_available') and not self.display_available:
            return
        
        self.screen.fill((0, 0, 0))
        
        # Draw grid with real map data
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                cell_type = self.grid[y, x]
                color = self.colors.get(cell_type, (0, 0, 0))
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
        
        # Draw additional real map features
        self._draw_map_features()
        
        # Draw agents and civilians
        super().render()
    
    def _draw_map_features(self):
        """Draw additional real map features"""
        if not self.node_positions:
            return
        
        # Draw nodes
        for node_id, (x, y) in self.node_positions.items():
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, (0, 255, 0), (center_x, center_y), 3)
        
        # Update display if available
        if hasattr(self, 'display_available') and self.display_available:
            pygame.display.flip()