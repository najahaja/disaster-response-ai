"""
Experience Replay Buffer for Reinforcement Learning
Key Features: Efficient sampling, prioritized experience replay support
"""

import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer:
    """Standard experience replay buffer"""
    def __init__(self, capacity=10000, obs_dim=4, action_dim=2, n_agents=1):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, states, actions, rewards, next_states, dones):
        """Add experience to buffer"""
        experience = {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.int64),
            'rewards': np.array(rewards, dtype=np.float32),
            'next_states': np.array(next_states, dtype=np.float32),
            'dones': np.array(dones, dtype=np.float32)
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample random batch from buffer"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        
        # Stack experiences
        batch_dict = {
            'states': np.stack([exp['states'] for exp in batch]),
            'actions': np.stack([exp['actions'] for exp in batch]),
            'rewards': np.stack([exp['rewards'] for exp in batch]),
            'next_states': np.stack([exp['next_states'] for exp in batch]),
            'dones': np.stack([exp['dones'] for exp in batch])
        }
        
        return batch_dict
    
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def push(self, state, action, reward, next_state, done):
        """Add experience with max priority"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample batch with probabilities proportional to priorities"""
        if self.size < batch_size:
            return None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get batch
        batch = []
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            batch.append((state, action, reward, next_state, done, idx, weights[idx]))
        
        return self._organize_batch(batch)
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small constant to avoid zero
    
    def _organize_batch(self, batch):
        """Organize batch into dictionaries"""
        states, actions, rewards, next_states, dones, indices, weights = zip(*batch)
        
        return {
            'states': torch.FloatTensor(np.array(states)),
            'actions': torch.LongTensor(np.array(actions)),
            'rewards': torch.FloatTensor(np.array(rewards)),
            'next_states': torch.FloatTensor(np.array(next_states)),
            'dones': torch.FloatTensor(np.array(dones)),
            'indices': torch.LongTensor(np.array(indices)),
            'weights': torch.FloatTensor(np.array(weights))
        }
    
    def __len__(self):
        return self.size

class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent environments"""
    def __init__(self, capacity=10000, obs_dim=4, action_dim=2, n_agents=2):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.buffers = [ReplayBuffer(capacity, obs_dim, action_dim) for _ in range(n_agents)]
    
    def push(self, states, actions, rewards, next_states, dones):
        """Push experience for each agent"""
        for i in range(self.n_agents):
            self.buffers[i].push(
                states[i] if len(states.shape) > 1 else states,
                actions[i] if len(actions.shape) > 1 else actions,
                rewards[i] if len(rewards.shape) > 1 else rewards,
                next_states[i] if len(next_states.shape) > 1 else next_states,
                dones[i] if len(dones.shape) > 1 else dones
            )
    
    def sample(self, batch_size):
        """Sample batch from all agents"""
        # Sample from each agent's buffer
        agent_batches = []
        for buffer in self.buffers:
            batch = buffer.sample(batch_size)
            if batch is None:
                return None
            agent_batches.append(batch)
        
        # Combine agent experiences
        combined_batch = {}
        for key in ['states', 'actions', 'rewards', 'next_states', 'dones']:
            combined = np.stack([batch[key] for batch in agent_batches], axis=1)
            combined_batch[key] = combined
        
        return combined_batch
    
    def __len__(self):
        return min(len(buffer) for buffer in self.buffers)