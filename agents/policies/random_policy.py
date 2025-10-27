import random

class RandomPolicy:
    """
    Policy for random agent movement (for testing)
    """
    
    def __init__(self):
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
    
    def get_action(self, agent_id, observation):
        """Get random action for an agent"""
        return random.choice(self.actions)
    
    def get_actions(self, agent_ids, observation):
        """Get actions for multiple agents"""
        return {agent_id: self.get_action(agent_id, observation) 
                for agent_id in agent_ids}