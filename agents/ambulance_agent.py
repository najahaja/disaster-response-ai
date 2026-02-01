from .base_agent import BaseAgent

class AmbulanceAgent(BaseAgent):
    """
    Ambulance agent for transporting multiple civilians
    
    Movement Restrictions:
    - Ground vehicle - CANNOT move through buildings
    - Can only move on: roads, hospitals, open spaces, collapsed buildings
    - Higher capacity than drones for civilian transport
    """
    
    def __init__(self, agent_id, position, config):
        super().__init__(agent_id, position, 'ambulance', config)
        self.civilians_on_board = 0
    
    def load_civilian(self):
        """
        Load a civilian onto the ambulance
        Returns: Boolean indicating if load was successful
        """
        if self.civilians_on_board < self.capacity:
            self.civilians_on_board += 1
            return True
        return False
    
    def unload_civilians(self):
        """
        Unload all civilians (typically at hospital)
        Returns: Number of civilians unloaded
        """
        unloaded_count = self.civilians_on_board
        self.civilians_rescued += unloaded_count
        self.civilians_on_board = 0
        return unloaded_count
    
    def get_state(self):
        """Return ambulance-specific state"""
        state = super().get_state()
        state['civilians_on_board'] = self.civilians_on_board
        state['available_capacity'] = self.capacity - self.civilians_on_board
        return state
    
    def __str__(self):
        return f"{super().__str__()} [Carrying: {self.civilians_on_board}/{self.capacity}]"