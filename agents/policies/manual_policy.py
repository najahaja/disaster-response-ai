class ManualPolicy:
    """
    Policy for manual control of agents (for testing)
    """
    
    def __init__(self):
        self.agent_controls = {}
    
    def set_control(self, agent_id, action):
        """Set control for a specific agent"""
        self.agent_controls[agent_id] = action
    
    def get_action(self, agent_id, observation):
        """Get action for an agent"""
        return self.agent_controls.get(agent_id, 'STAY')
    
    def clear_controls(self):
        """Clear all controls"""
        self.agent_controls = {}
    
    def get_available_actions(self):
        """Get list of available actions"""
        return ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']