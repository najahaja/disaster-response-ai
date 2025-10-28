import pettingzoo
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces
import numpy as np
import pygame
from typing import Dict, List, Optional, Any

# Import from your project
from environments.simple_grid_env import SimpleGridEnv

# Try to import MARL components with fallbacks
try:
    from .reward_functions import CollaborativeReward
    from .observation_spaces import GlobalObservation
    from .action_spaces import DiscreteActionSpace
except ImportError:
    # Fallback imports if relative imports fail
    try:
        from marl.reward_functions import CollaborativeReward
        from marl.observation_spaces import GlobalObservation
        from marl.action_spaces import DiscreteActionSpace
    except ImportError:
        # Create minimal fallback classes
        class CollaborativeReward:
            def __init__(self, config):
                self.config = config
            def calculate_reward(self, agent, prev_obs, action, next_obs):
                return 0.0
        
        class GlobalObservation:
            def __init__(self, grid_size, num_cell_types, num_agent_types):
                self.grid_size = grid_size
            def encode_observation(self, agent, observation):
                return np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            def get_observation_space(self, agent_type, grid_size):
                return spaces.Box(low=0, high=255, shape=(grid_size, grid_size, 3), dtype=np.uint8)
        
        class DiscreteActionSpace:
            def __init__(self):
                self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'REST']
            def decode_action(self, action, agent):
                return self.actions[action] if 0 <= action < len(self.actions) else 'STAY'
            def get_action_space(self, agent_type):
                return spaces.Discrete(6)

class DisasterResponseEnv(AECEnv):
    """
    PettingZoo wrapper for the Disaster Response environment
    Fixed version with proper error handling
    """
    
    metadata = {
        "name": "disaster_response_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10
    }
    
    def __init__(self, config_path="config.yaml", render_mode=None):
        super().__init__()
        
        # Initialize the underlying environment
        self.env = SimpleGridEnv(config_path)
        self.config = self.env.config
        self.render_mode = render_mode
        
        # Initialize MARL components
        self.reward_function = CollaborativeReward(self.config)
        self.observation_space_obj = GlobalObservation(
            self.env.grid_size,
            len(self.env.cell_types) if hasattr(self.env, 'cell_types') else 4,
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
        
        # Track cumulative rewards for PettingZoo compliance
        self._cumulative_rewards = {}
        
        # Initialize the environment
        self.reset()
    
    @property
    def observation_spaces(self):
        """Property required by PettingZoo"""
        return {agent: self.observation_space(agent) for agent in self.possible_agents}
    
    @property
    def action_spaces(self):
        """Property required by PettingZoo"""
        return {agent: self.action_space(agent) for agent in self.possible_agents}
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Reset underlying environment
        obs, info = self.env.reset()
        
        # Initialize agents list - create some default agents if none exist
        if not hasattr(self.env, 'agents') or not self.env.agents:
            # Add some default agents for testing
            from agents.drone_agent import DroneAgent
            from agents.ambulance_agent import AmbulanceAgent
            
            drone = DroneAgent("drone_0", np.array([5, 5]), self.config)
            ambulance = AmbulanceAgent("ambulance_0", np.array([10, 10]), self.config)
            
            self.env.add_agent(drone)
            self.env.add_agent(ambulance)
        
        self.possible_agents = list(self.env.agents.keys())
        self.agents = self.possible_agents[:]
        
        # Initialize agent selector
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        # Initialize state dictionaries
        self.observations = {}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Get initial observations
        for agent_id in self.agents:
            agent = self.env.agents[agent_id]
            self.observations[agent_id] = self.observation_space_obj.encode_observation(
                agent, obs
            )
            self.infos[agent_id] = info
        
        return self.observations
    
    def step(self, action):
        """
        Step the environment for the current agent
        """
        # Handle dead agents
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        
        current_agent = self.agent_selection
        agent = self.env.agents[current_agent]
        
        # Decode action
        action_str = self.action_space_obj.decode_action(action, agent)
        
        # Store current state for reward calculation
        prev_observation = self.env.get_observation()
        
        # Execute action for current agent only
        actions = {current_agent: action_str}
        
        # Handle both gymnasium and legacy step formats
        result = self.env.step(actions)
        
        if len(result) == 5:
            # Gymnasium format: obs, reward, terminated, truncated, info
            next_observation, single_reward, terminated, truncated, step_info = result
            done = terminated or truncated
            
            # Convert single reward to multi-agent format
            # Distribute reward among agents or assign to current agent
            if isinstance(single_reward, (int, float)):
                rewards = {current_agent: single_reward}
                for agent_id in self.agents:
                    if agent_id != current_agent:
                        rewards[agent_id] = 0.0
            else:
                rewards = {current_agent: single_reward}
                
        elif len(result) == 4:
            # Legacy format: obs, rewards, done, info
            next_observation, rewards, done, step_info = result
        else:
            raise ValueError(f"Unexpected step result format: {len(result)} values")
        
        # Calculate reward using reward function (if available)
        try:
            individual_reward = self.reward_function.calculate_reward(
                agent, prev_observation, action_str, next_observation
            )
        except:
            # Fallback: use the reward from the environment
            individual_reward = rewards.get(current_agent, 0.0)
        
        # Update all agents' states
        for agent_id in self.agents:
            agent_obj = self.env.agents[agent_id]
            
            # Update observation
            self.observations[agent_id] = self.observation_space_obj.encode_observation(
                agent_obj, next_observation
            )
            
            # Update reward (use individual reward for current agent, 0 for others in this step)
            if agent_id == current_agent:
                self.rewards[agent_id] = individual_reward
            else:
                self.rewards[agent_id] = 0.0
                
            # Update info
            self.infos[agent_id] = step_info
            
            # Update termination status
            self.terminations[agent_id] = done
            self.truncations[agent_id] = done
        
        # Accumulate rewards for PettingZoo
        self._accumulate_rewards()
        
        # Move to next agent if not done
        if not done:
            self.agent_selection = self._agent_selector.next()
    
    def _accumulate_rewards(self):
        """Accumulate rewards for PettingZoo compliance"""
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
    
    def observe(self, agent):
        """
        Get observation for a specific agent
        """
        return self.observations.get(agent, np.zeros((self.env.grid_size, self.env.grid_size, 3), dtype=np.uint8))
    
    def observation_space(self, agent):
        """
        Get observation space for a specific agent
        """
        try:
            agent_type = self.env.agents[agent].agent_type
        except:
            agent_type = "drone"  # Default fallback
            
        return self.observation_space_obj.get_observation_space(
            agent_type, self.env.grid_size
        )
    
    def action_space(self, agent):
        """
        Get action space for a specific agent
        """
        try:
            agent_type = self.env.agents[agent].agent_type
        except:
            agent_type = "drone"  # Default fallback
            
        return self.action_space_obj.get_action_space(agent_type)
    
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
            state_components.append(agent_obs.flatten())
        
        if state_components:
            return np.concatenate(state_components)
        else:
            return np.array([])
    
    def _was_dead_step(self, action):
        """
        Handle steps for terminated agents - FIXED VERSION
        """
        current_agent = self.agent_selection
        
        # Set reward to 0 for dead agent
        self.rewards[current_agent] = 0
        
        # Move to next agent
        self.agent_selection = self._agent_selector.next()
        
        # Clear rewards for the next step
        self._clear_rewards()

# ✅ FIXED: Create callable functions/classes for exports
class DisasterResponsePettingZoo(DisasterResponseEnv):
    """Alias for backward compatibility"""
    pass

def DisasterResponsePettingZooEnv(config_path="config.yaml", render_mode=None):
    """Create a PettingZoo-compatible environment - CALLABLE FUNCTION"""
    return DisasterResponseEnv(config_path, render_mode)

# Standard PettingZoo interface
def env(**kwargs):
    """
    PettingZoo environment creation function
    This is the standard PettingZoo interface
    """
    return DisasterResponseEnv(**kwargs)

def test_pettingzoo_wrapper():
    """Test function for the PettingZoo wrapper"""
    print("🧪 Testing PettingZoo wrapper...")
    
    try:
        # Create environment using the correct function
        env_instance = DisasterResponsePettingZooEnv()
        print("✅ Environment created successfully")
        
        # Test reset
        observations = env_instance.reset()
        print(f"✅ Reset successful. Agents: {env_instance.agents}")
        print(f"✅ Observations shape: {list(observations.values())[0].shape if observations else 'No observations'}")
        
        # Test step (if we have agents)
        if env_instance.agents:
            # Get first agent
            first_agent = env_instance.agents[0]
            env_instance.agent_selection = first_agent
            
            # Take a random action
            action_space = env_instance.action_space(first_agent)
            action = action_space.sample() if hasattr(action_space, 'sample') else 0
            
            env_instance.step(action)
            print(f"✅ Step successful for agent {first_agent}")
            
        # Test observation and action spaces
        for agent in env_instance.agents:
            obs_space = env_instance.observation_space(agent)
            act_space = env_instance.action_space(agent)
            print(f"✅ Agent {agent}: obs_space={type(obs_space)}, act_space={type(act_space)}")
        
        env_instance.close()
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

# ✅ FIXED: Define what gets imported
__all__ = [
    'DisasterResponseEnv', 
    'DisasterResponsePettingZoo',
    'DisasterResponsePettingZooEnv',
    'env',
    'test_pettingzoo_wrapper'
]

if __name__ == "__main__":
    test_pettingzoo_wrapper()