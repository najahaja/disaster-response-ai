"""
QMIX Model Implementation for Multi-Agent Reinforcement Learning
Key Features: Centralized training with decentralized execution, monotonic mixing network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import copy

class AgentNetwork(nn.Module):
    """Individual agent Q-network"""
    def __init__(self, obs_dim, action_dim, hidden_dim=64, rnn_hidden_dim=32):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.use_rnn = rnn_hidden_dim > 0
        
        # Feature extraction
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # RNN for temporal dependencies (optional)
        if self.use_rnn:
            self.rnn = nn.GRU(hidden_dim, rnn_hidden_dim, batch_first=True)
            self.fc3 = nn.Linear(rnn_hidden_dim, action_dim)
        else:
            self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Dueling architecture (optional)
        self.dueling = True
        if self.dueling:
            self.value_stream = nn.Linear(hidden_dim if not self.use_rnn else rnn_hidden_dim, 1)
            self.advantage_stream = nn.Linear(hidden_dim if not self.use_rnn else rnn_hidden_dim, action_dim)
    
    def forward(self, obs, hidden_state=None):
        """Forward pass for single agent"""
        batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
        
        # Feature extraction
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        
        # RNN processing if enabled
        if self.use_rnn:
            if hidden_state is None:
                hidden_state = torch.zeros(1, batch_size, self.rnn_hidden_dim).to(obs.device)
            x = x.unsqueeze(1) if len(x.shape) == 2 else x
            x, new_hidden = self.rnn(x, hidden_state)
            x = x.squeeze(1) if x.shape[1] == 1 else x
        
        # Dueling architecture or standard Q-values
        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.fc3(x)
        
        if self.use_rnn:
            return q_values, new_hidden
        return q_values, hidden_state

class HyperNetwork(nn.Module):
    """Hypernetwork for generating mixing weights from global state"""
    def __init__(self, state_dim, n_agents, mixing_hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.mixing_hidden_dim = mixing_hidden_dim
        
        # Hypernetwork for mixing weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, n_agents * mixing_hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, mixing_hidden_dim)
        )
        
        # Hypernetwork for biases
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)
    
    def forward(self, global_state):
        """Generate mixing weights and biases from global state"""
        # First layer weights and bias
        w1 = torch.abs(self.hyper_w1(global_state)).view(-1, self.n_agents, self.mixing_hidden_dim)
        b1 = self.hyper_b1(global_state).view(-1, 1, self.mixing_hidden_dim)
        
        # Second layer weights and bias
        w2 = torch.abs(self.hyper_w2(global_state)).view(-1, self.mixing_hidden_dim, 1)
        b2 = self.hyper_b2(global_state).view(-1, 1, 1)
        
        return w1, w2, b1, b2

class MixingNetwork(nn.Module):
    """Mixing network that combines individual Q-values"""
    def __init__(self, n_agents, state_dim, mixing_hidden_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_hidden_dim = mixing_hidden_dim
        self.hyper_network = HyperNetwork(state_dim, n_agents, mixing_hidden_dim)
    
    def forward(self, agent_qs, global_state):
        """Combine individual Q-values into joint Q-value with monotonicity constraint"""
        batch_size = agent_qs.shape[0]
        
        # Get mixing parameters from hypernetwork
        w1, w2, b1, b2 = self.hyper_network(global_state)
        
        # Ensure positive weights for monotonicity
        w1 = F.elu(w1) + 1
        w2 = F.elu(w2) + 1
        
        # First mixing layer
        agent_qs = agent_qs.view(batch_size, 1, -1)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Second mixing layer
        q_total = torch.bmm(hidden, w2) + b2
        
        return q_total.view(-1, 1)

class QMIXModel(nn.Module):
    """Complete QMIX model for multi-agent reinforcement learning"""
    def __init__(self, n_agents, obs_dim, action_dim, 
                 hidden_dim=64, mixing_hidden_dim=32, 
                 rnn_hidden_dim=32, gamma=0.99, 
                 use_rnn=True, double_q=True, dueling=True):
        super().__init__()
        
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mixing_hidden_dim = mixing_hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.gamma = gamma
        self.use_rnn = use_rnn
        self.double_q = double_q
        self.dueling = dueling
        
        # Individual agent networks
        self.agent_networks = nn.ModuleList([
            AgentNetwork(obs_dim, action_dim, hidden_dim, rnn_hidden_dim if use_rnn else 0)
            for _ in range(n_agents)
        ])
        
        # Mixing network
        self.mixing_network = MixingNetwork(n_agents, obs_dim * n_agents, mixing_hidden_dim)
        
        # Target networks
        self.target_agent_networks = nn.ModuleList([
            AgentNetwork(obs_dim, action_dim, hidden_dim, rnn_hidden_dim if use_rnn else 0)
            for _ in range(n_agents)
        ])
        self.target_mixing_network = MixingNetwork(n_agents, obs_dim * n_agents, mixing_hidden_dim)
        
        # Initialize target networks
        self.update_target_networks(tau=1.0)
        
        # Training parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update_freq = 200
        self.train_step_counter = 0
    
    def forward(self, obs, hidden_states=None):
        """Forward pass for all agents"""
        batch_size = obs.shape[0] if len(obs.shape) > 2 else 1
        
        if hidden_states is None and self.use_rnn:
            hidden_states = [None] * self.n_agents
        
        agent_qs = []
        new_hidden_states = []
        
        # Get Q-values from each agent
        for i in range(self.n_agents):
            agent_obs = obs[:, i, :] if len(obs.shape) > 2 else obs[i, :].unsqueeze(0)
            
            if self.use_rnn:
                q_values, new_hidden = self.agent_networks[i](agent_obs, hidden_states[i])
                new_hidden_states.append(new_hidden)
            else:
                q_values, _ = self.agent_networks[i](agent_obs, None)
            
            agent_qs.append(q_values)
        
        # Stack agent Q-values
        agent_qs = torch.stack(agent_qs, dim=1)  # [batch, n_agents, action_dim]
        
        if self.use_rnn:
            return agent_qs, new_hidden_states
        return agent_qs, None
    
    def get_actions(self, obs, hidden_states=None, training=True):
        """Select actions using epsilon-greedy policy"""
        batch_size = obs.shape[0] if len(obs.shape) > 2 else 1
        
        with torch.no_grad():
            if self.use_rnn:
                agent_qs, new_hidden_states = self.forward(obs, hidden_states)
            else:
                agent_qs, _ = self.forward(obs, hidden_states)
                new_hidden_states = None
        
        actions = []
        
        # Epsilon-greedy action selection
        for i in range(self.n_agents):
            if training and np.random.random() < self.epsilon:
                # Random action
                action = torch.randint(0, self.action_dim, (batch_size,))
            else:
                # Greedy action
                q_values = agent_qs[:, i, :] if len(agent_qs.shape) > 2 else agent_qs[i, :]
                action = torch.argmax(q_values, dim=-1)
            
            actions.append(action)
        
        actions = torch.stack(actions, dim=1)
        
        return actions, new_hidden_states
    
    def compute_total_q(self, agent_qs, global_state):
        """Compute total Q-value using mixing network"""
        batch_size = agent_qs.shape[0]
        n_agents = agent_qs.shape[1]
        
        # Reshape for mixing
        agent_qs = agent_qs.view(batch_size, n_agents, -1)
        
        # Get max Q-value per agent
        max_qs = torch.max(agent_qs, dim=-1)[0]
        
        # Mix Q-values
        total_q = self.mixing_network(max_qs, global_state)
        
        return total_q
    
    def update_target_networks(self, tau=0.01):
        """Soft update of target networks"""
        for target_net, net in zip(self.target_agent_networks, self.agent_networks):
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_mixing_network.parameters(), 
                                      self.mixing_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def train_step(self, batch, optimizer):
        """Single training step"""
        states = batch['states'].float()
        actions = batch['actions'].long()
        rewards = batch['rewards'].float()
        next_states = batch['next_states'].float()
        dones = batch['dones'].float()
        
        batch_size = states.shape[0]
        
        # Current Q-values
        agent_qs, _ = self.forward(states)
        current_qs = agent_qs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute total current Q
        global_state = states.view(batch_size, -1)
        current_total_q = self.compute_total_q(current_qs, global_state)
        
        # Next Q-values with target network
        with torch.no_grad():
            if self.double_q:
                # Double Q-learning
                next_agent_qs, _ = self.forward(next_states)
                next_actions = torch.argmax(next_agent_qs, dim=-1)
                
                next_target_qs, _ = self.target_forward(next_states)
                next_max_qs = next_target_qs.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
            else:
                # Standard Q-learning
                next_target_qs, _ = self.target_forward(next_states)
                next_max_qs = torch.max(next_target_qs, dim=-1)[0]
            
            # Compute total next Q
            next_global_state = next_states.view(batch_size, -1)
            next_total_q = self.target_compute_total_q(next_max_qs, next_global_state)
            
            # Compute target
            target_q = rewards.sum(dim=1, keepdim=True) + self.gamma * next_total_q * (1 - dones[:, 0:1])
        
        # Compute loss
        loss = F.mse_loss(current_total_q, target_q)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        optimizer.step()
        
        # Update target networks periodically
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_networks(tau=0.01)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def target_forward(self, obs, hidden_states=None):
        """Forward pass with target networks"""
        batch_size = obs.shape[0] if len(obs.shape) > 2 else 1
        
        if hidden_states is None and self.use_rnn:
            hidden_states = [None] * self.n_agents
        
        agent_qs = []
        new_hidden_states = []
        
        for i in range(self.n_agents):
            agent_obs = obs[:, i, :] if len(obs.shape) > 2 else obs[i, :].unsqueeze(0)
            
            if self.use_rnn:
                q_values, new_hidden = self.target_agent_networks[i](agent_obs, hidden_states[i])
                new_hidden_states.append(new_hidden)
            else:
                q_values, _ = self.target_agent_networks[i](agent_obs, None)
            
            agent_qs.append(q_values)
        
        agent_qs = torch.stack(agent_qs, dim=1)
        
        if self.use_rnn:
            return agent_qs, new_hidden_states
        return agent_qs, None
    
    def target_compute_total_q(self, agent_qs, global_state):
        """Compute total Q-value using target mixing network"""
        batch_size = agent_qs.shape[0]
        n_agents = agent_qs.shape[1]
        
        agent_qs = agent_qs.view(batch_size, n_agents, -1)
        max_qs = torch.max(agent_qs, dim=-1)[0]
        
        total_q = self.target_mixing_network(max_qs, global_state)
        
        return total_q
    
    def init_hidden(self, batch_size=1):
        """Initialize RNN hidden states"""
        if not self.use_rnn:
            return None
        
        hidden_states = []
        for _ in range(self.n_agents):
            hidden = torch.zeros(1, batch_size, self.rnn_hidden_dim)
            hidden_states.append(hidden)
        
        return hidden_states
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'n_agents': self.n_agents,
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'mixing_hidden_dim': self.mixing_hidden_dim,
                'rnn_hidden_dim': self.rnn_hidden_dim,
                'gamma': self.gamma,
                'use_rnn': self.use_rnn,
                'double_q': self.double_q,
                'dueling': self.dueling
            },
            'epsilon': self.epsilon
        }, path)
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.epsilon = checkpoint.get('epsilon', 0.05)
        
        return model