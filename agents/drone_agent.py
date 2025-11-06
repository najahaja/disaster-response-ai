from .base_agent import BaseAgent
import numpy as np

class DroneAgent(BaseAgent):
    """
    Drone agent with scouting capabilities
    """
    
    def __init__(self, agent_id, position, config):
        super().__init__(agent_id, position, 'drone', config)
        self.scouted_locations = set()
        self.scout_range = 3  # Cells around drone that can be scouted
    
    
    
    def _scout_area(self, grid):
        """Scout the area around the drone"""
        center_x, center_y = self.position
        for dx in range(-self.scout_range, self.scout_range + 1):
            for dy in range(-self.scout_range, self.scout_range + 1):
                scout_x, scout_y = center_x + dx, center_y + dy
                if (0 <= scout_x < grid.shape[0] and 
                    0 <= scout_y < grid.shape[1]):
                    self.scouted_locations.add((scout_x, scout_y))
    
    def is_location_scouted(self, position):
        """Check if a location has been scouted"""
        return tuple(position) in self.scouted_locations
    
    def get_scouted_locations(self):
        """Get all scouted locations"""
        return list(self.scouted_locations)
    
    def get_state(self):
        """Return drone-specific state"""
        state = super().get_state()
        state['scouted_locations_count'] = len(self.scouted_locations)
        state['scout_range'] = self.scout_range
        return state