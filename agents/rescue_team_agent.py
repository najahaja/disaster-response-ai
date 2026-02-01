from .base_agent import BaseAgent

class RescueTeamAgent(BaseAgent):
    """
    Rescue team agent for complex rescue operations
    
    Movement Restrictions:
    - Ground vehicle - CANNOT move through buildings
    - Can only move on: roads, hospitals, open spaces, collapsed buildings
    - Specialized in rescuing civilians from collapsed buildings
    """
    
    def __init__(self, agent_id, position, config):
        super().__init__(agent_id, position, 'rescue_team', config)
        self.operation_time = 0  # Time spent on current operation
        self.complex_operations = 0
    
    def start_complex_operation(self):
        """Start a complex rescue operation"""
        self.operation_time = 0
        return True
    
    def update_operation(self):
        """Update the ongoing operation"""
        if self.operation_time > 0:
            self.operation_time += 1
    
    def complete_operation(self):
        """Complete the current operation"""
        if self.operation_time > 0:
            self.complex_operations += 1
            success = self.rescue_civilian()  # Rescue civilian after operation
            self.operation_time = 0
            return success
        return False
    
    def get_state(self):
        """Return rescue team-specific state"""
        state = super().get_state()
        state['complex_operations'] = self.complex_operations
        state['current_operation_time'] = self.operation_time
        return state
    
    def __str__(self):
        return f"{super().__str__()} [Operations: {self.complex_operations}]"