import pettingzoo
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector  # ✅ FIXED IMPORT
from gymnasium import spaces
import numpy as np
import pygame
from typing import Dict, List, Optional

from environments.simple_grid_env import SimpleGridEnv
from .reward_functions import CollaborativeReward
from .observation_spaces import GlobalObservation
from .action_spaces import DiscreteActionSpace

class DisasterResponseEnv(AECEnv):
    """
    PettingZoo wrapper for the Disaster Response environment
    """
    
    metadata = {
        "name": "disaster_response_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10
    }
    
    def __init__(self, config_path="config.yaml", render_mode="human"):
        super().__init__()
        
        # Initialize the underlying environment
        self.env = SimpleGridEnv(config_path)
        self.config = self.env.config
        self.render_mode = render_mode
        
        # Initialize MARL components
        self.reward_function = CollaborativeReward(self.config)
        self.observation_space_obj = GlobalObservation(
            self.env.grid_size,
            len(self.env.cell_types),
            3  # num agent types
        )
        self.action_space_obj = DiscreteActionSpace()
        
        # PettingZoo required attributes
        self.possible_agents = []
        self.agents = []
        self._agent_selector = None
        
        # Environment state
        self.observations = {}
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        
        # Initialize the environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment
        """
        # Reset underlying environment
        observation = self.env.reset()
        
        # Initialize agents list
        self.possible_agents = list(self.env.agents.keys())
        self.agents = self.possible_agents[:]
        
        # ✅ FIXED: Use AgentSelector correctly
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        # Initialize state dictionaries
        self.observations = {}
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Get initial observations
        for agent_id in self.agents:
            agent = self.env.agents[agent_id]
            self.observations[agent_id] = self.observation_space_obj.encode_observation(
                agent, observation
            )
        
        return self.observations
    
    def step(self, action):
        """
        Step the environment for the current agent
        """
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        
        agent = self.env.agents[self.agent_selection]
        
        # Decode action
        action_str = self.action_space_obj.decode_action(action, agent)
        
        # Store current state for reward calculation
        prev_observation = self.env.get_observation()
        
        # Execute action for current agent only
        actions = {self.agent_selection: action_str}
        next_observation, rewards, done, info = self.env.step(actions)
        
        # Calculate reward using reward function
        individual_reward = self.reward_function.calculate_reward(
            agent, prev_observation, action_str, next_observation
        )
        
        # Update agent's total reward
        agent.update_reward(individual_reward)
        
        # Store results
        self.rewards[self.agent_selection] = individual_reward
        self.observations[self.agent_selection] = self.observation_space_obj.encode_observation(
            agent, next_observation
        )
        self.infos[self.agent_selection] = info
        
        # Check if episode is done
        if done:
            self.terminations = {agent: True for agent in self.agents}
            self.truncations = {agent: True for agent in self.agents}
        else:
            # Move to next agent
            self.agent_selection = self._agent_selector.next()
    
    def observe(self, agent):
        """
        Get observation for a specific agent
        """
        return self.observations.get(agent, None)
    
    def observation_space(self, agent):
        """
        Get observation space for a specific agent
        """
        return self.observation_space_obj.get_observation_space(
            self.env.agents[agent].agent_type, self.env.grid_size
        )
    
    def action_space(self, agent):
        """
        Get action space for a specific agent
        """
        return self.action_space_obj.get_action_space(
            self.env.agents[agent].agent_type
        )
    
    def render(self):
        """
        Render the environment
        """
        return self.env.render()
    
    def close(self):
        """
        Close the environment
        """
        self.env.close()
    
    def state(self):
        """
        Get the global state (for centralized training)
        """
        observation = self.env.get_observation()
        
        # Combine all agent observations
        state_components = []
        for agent_id in self.agents:
            agent = self.env.agents[agent_id]
            agent_obs = self.observation_space_obj.encode_observation(agent, observation)
            state_components.append(agent_obs)
        
        return np.concatenate(state_components)
    
    def _was_dead_step(self, action):
        """
        Handle steps for terminated agents
        """
        self.rewards[self.agent_selection] = 0
        self.terminations[self.agent_selection] = True
        self.agent_selection = self._agent_selector.next()