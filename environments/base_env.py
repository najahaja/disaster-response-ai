from abc import ABC, abstractmethod
import numpy as np

class BaseEnvironment(ABC):
    """
    Abstract base class for all disaster response environments
    """
    
    def __init__(self, config):
        self.config = config
        self.agents = {}
        self.step_count = 0
        self.max_steps = config['environment']['max_steps']
        self.disaster_triggered = False
        
    @abstractmethod
    def reset(self):
        """Reset the environment to initial state"""
        pass
        
    @abstractmethod 
    def step(self, actions):
        """
        Execute one time step
        Args:
            actions: Dictionary of {agent_id: action}
        Returns:
            observation, rewards, done, info
        """
        pass
        
    @abstractmethod
    def render(self):
        """Render the environment"""
        pass
        
    @abstractmethod
    def get_observation(self):
        """Get current observation of the environment"""
        pass
        
    @abstractmethod
    def calculate_rewards(self):
        """Calculate rewards for each agent"""
        pass
        
    @abstractmethod
    def trigger_disaster(self):
        """Trigger disaster scenario"""
        pass
        
    def is_done(self):
        """Check if episode is done"""
        return self.step_count >= self.max_steps
        
    def add_agent(self, agent):
        """Add an agent to the environment"""
        self.agents[agent.agent_id] = agent
        
    def remove_agent(self, agent_id):
        """Remove an agent from the environment"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            
    def get_agent_positions(self):
        """Get positions of all agents"""
        return {aid: agent.position for aid, agent in self.agents.items()}
        
    def __str__(self):
        return f"{self.__class__.__name__}(agents={len(self.agents)}, steps={self.step_count})"