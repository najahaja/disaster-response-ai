from abc import ABC, abstractmethod
import numpy as np

class RewardFunction(ABC):
    """Abstract base class for reward functions"""
    
    @abstractmethod
    def calculate_reward(self, agent, env_observation, action, next_observation):
        """Calculate reward for an agent"""
        pass
    
    @abstractmethod
    def calculate_global_reward(self, agents, env_observation):
        """Calculate global reward for the system"""
        pass

class CollaborativeReward(RewardFunction):
    """Collaborative reward function encouraging teamwork"""
    
    def __init__(self, config):
        self.config = config
        self.cell_types = config['environment']['cell_types']
        
        # Reward weights
        self.civilian_rescued_reward = 100
        self.step_penalty = -1
        self.collaboration_bonus = 20
        self.efficiency_bonus = 10
        self.blocked_penalty = -5
        
    def calculate_reward(self, agent, env_observation, action, next_observation):
        """Calculate individual agent reward"""
        reward = 0
        
        # Step penalty (encourage efficiency)
        reward += self.step_penalty
        
        # Check if agent rescued a civilian
        prev_civs = agent.civilians_rescued
        current_civs = agent.civilians_rescued
        if current_civs > prev_civs:
            reward += self.civilian_rescued_reward
        
        # Efficiency bonus for drones scouting new areas
        if hasattr(agent, 'scouted_locations'):
            new_scouts = len(agent.scouted_locations)
            if hasattr(agent, '_prev_scout_count'):
                if new_scouts > agent._prev_scout_count:
                    reward += self.efficiency_bonus
            agent._prev_scout_count = new_scouts
        
        # Penalty for moving into blocked areas
        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            new_pos = agent._calculate_new_position(action)
            grid = env_observation['grid']
            if (0 <= new_pos[0] < grid.shape[0] and 
                0 <= new_pos[1] < grid.shape[1] and 
                grid[new_pos[1], new_pos[0]] == self.cell_types['BLOCKED']):
                reward += self.blocked_penalty
        
        return reward
    
    def calculate_global_reward(self, agents, env_observation):
        """Calculate global reward based on team performance"""
        total_reward = 0
        
        # Base reward from individual actions
        for agent in agents.values():
            total_reward += agent.total_reward
        
        # Collaboration bonus - reward when agents work together
        civilian_positions = [civ['position'] for civ in env_observation['civilians'] 
                            if not civ['rescued']]
        
        for civ_pos in civilian_positions:
            agents_near_civilian = 0
            for agent in agents.values():
                distance = np.linalg.norm(np.array(agent.position) - np.array(civ_pos))
                if distance <= 2:  # Agents within 2 cells
                    agents_near_civilian += 1
            
            if agents_near_civilian >= 2:
                total_reward += self.collaboration_bonus
        
        return total_reward

class CompetitiveReward(RewardFunction):
    """Competitive reward function - agents compete to rescue civilians"""
    
    def __init__(self, config):
        self.config = config
        self.civilian_rescued_reward = 150  # Higher for competition
        self.step_penalty = -1
        self.duplicate_penalty = -50  # Penalty for multiple agents going for same civilian
    
    def calculate_reward(self, agent, env_observation, action, next_observation):
        reward = 0
        reward += self.step_penalty
        
        # Check for civilian rescue
        prev_civs = agent.civilians_rescued
        current_civs = agent.civilians_rescued
        if current_civs > prev_civs:
            reward += self.civilian_rescued_reward
        
        # Penalty for multiple agents targeting same civilian
        civ_positions = [civ['position'] for civ in env_observation['civilians'] 
                        if not civ['rescued']]
        
        for civ_pos in civ_positions:
            agents_targeting = 0
            for other_agent in env_observation['agents'].values():
                if (isinstance(other_agent, dict) and 
                    np.array_equal(other_agent['position'], civ_pos)):
                    agents_targeting += 1
                elif (hasattr(other_agent, 'position') and 
                      np.array_equal(other_agent.position, civ_pos)):
                    agents_targeting += 1
            
            if agents_targeting > 1:
                reward += self.duplicate_penalty
        
        return reward
    
    def calculate_global_reward(self, agents, env_observation):
        # In competitive mode, global reward is just sum of individual rewards
        return sum(agent.total_reward for agent in agents.values())