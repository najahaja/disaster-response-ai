"""
Proximal Policy Optimization (PPO) Model Implementation
Key Features: Clipped surrogate objective, actor-critic architecture, GAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal
import copy

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_layers=2, 
                 continuous=False, action_std_init=0.6):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.continuous = continuous
        
        # Shared feature extractor
        self.shared_layers = nn.ModuleList()
        input_dim = state_dim
        
        for i in range(n_layers):
            self.shared_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        # Actor network (policy)
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.ones(action_dim) * np.log(action_std_init))
        else:
            self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic network (value function)
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """Forward pass through network"""
        x = state
        
        # Shared layers
        for layer in self.shared_layers:
            x = F.relu(layer(x))
        
        # Actor output
        if self.continuous:
            action_mean = self.actor_mean(x)
            action_log_std = self.actor_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            action_dist = Normal(action_mean, action_std)
        else:
            action_logits = self.actor_head(x)
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = Categorical(action_probs)
        
        # Critic output
        state_value = self.critic_head(x)
        
        return action_dist, state_value
    
    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        with torch.no_grad():
            action_dist, state_value = self.forward(state)
            
            if deterministic:
                if self.continuous:
                    action = action_dist.mean
                else:
                    action = torch.argmax(action_dist.probs, dim=-1)
            else:
                action = action_dist.sample()
            
            action_logprob = action_dist.log_prob(action)
            
            return action, action_logprob, state_value
    
    def evaluate_actions(self, states, actions):
        """Evaluate actions for given states"""
        action_dist, state_values = self.forward(states)
        
        if self.continuous:
            # For continuous actions, actions might need reshaping
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(-1)
        else:
            # For discrete actions
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(-1)
        
        action_logprobs = action_dist.log_prob(actions)
        dist_entropy = action_dist.entropy().mean()
        
        return action_logprobs, state_values, dist_entropy

class PPOModel:
    """Complete PPO implementation with training logic"""
    def __init__(self, state_dim, action_dim, n_agents=1, 
                 hidden_dim=128, n_layers=2, continuous=False,
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.continuous = continuous
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.policy_net = ActorCriticNetwork(
            state_dim, action_dim, hidden_dim, n_layers, continuous
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=0.0003,  # Default PPO learning rate
            eps=1e-5
        )
        
        # Training state
        self.epoch = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
    
    def compute_gae(self, rewards, values, dones, next_value=None):
        """
        Compute Generalized Advantage Estimation
        """
        batch_size = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # Handle next_value for terminal states
        if next_value is None:
            next_value = values[-1]
        
        # Compute advantages in reverse
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            # Temporal difference error
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            
            # GAE
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, batch, n_epochs=4, batch_size=64):
        """
        Update policy using PPO clipped objective
        """
        # Unpack batch
        states = torch.FloatTensor(batch['states'])
        actions = torch.FloatTensor(batch['actions']) if self.continuous else torch.LongTensor(batch['actions'])
        old_logprobs = torch.FloatTensor(batch['logprobs'])
        rewards = torch.FloatTensor(batch['rewards'])
        dones = torch.FloatTensor(batch['dones'])
        values = torch.FloatTensor(batch['values'])
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Convert to tensors
        advantages = advantages.detach()
        returns = returns.detach()
        
        # Training statistics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kl_divs = []
        clip_fractions = []
        
        # Multiple epochs of optimization
        for epoch in range(n_epochs):
            # Create random indices for mini-batch sampling
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                new_logprobs, state_values, dist_entropy = self.policy_net.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Reshape if needed
                if len(new_logprobs.shape) == 2:
                    new_logprobs = new_logprobs.squeeze(-1)
                if len(batch_old_logprobs.shape) == 2:
                    batch_old_logprobs = batch_old_logprobs.squeeze(-1)
                
                # Ratio
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(state_values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -dist_entropy
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                
                # Track statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # KL divergence approximation
                with torch.no_grad():
                    approx_kl = ((batch_old_logprobs - new_logprobs).mean()).item()
                    approx_kl_divs.append(approx_kl)
                    
                    # Clip fraction
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    clip_fractions.append(clip_fraction)
        
        # Update epoch counter
        self.epoch += 1
        
        # Return statistics
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'approx_kl': np.mean(approx_kl_divs),
            'clip_fraction': np.mean(clip_fractions),
            'epoch': self.epoch
        }
        
        return stats
    
    def get_action(self, state, deterministic=False):
        """Get action for given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, logprob, value = self.policy_net.get_action(state_tensor, deterministic)
            
            # Convert to numpy
            action = action.squeeze(0).cpu().numpy()
            logprob = logprob.squeeze(0).cpu().numpy()
            value = value.squeeze(0).cpu().numpy()
            
            return action, logprob, value
    
    def save(self, path, stats=None):
        """Save model checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'n_agents': self.n_agents,
                'hidden_dim': self.hidden_dim,
                'n_layers': self.n_layers,
                'continuous': self.continuous,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef
            }
        }
        
        if stats:
            checkpoint['stats'] = stats
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        
        # Load state dicts
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.epoch = checkpoint['epoch']
        self.total_steps = checkpoint['total_steps']
        self.best_reward = checkpoint['best_reward']
        
        print(f"Model loaded from {path} (epoch {self.epoch})")
        return checkpoint.get('stats', None)

class MultiAgentPPO:
    """Multi-agent extension of PPO"""
    def __init__(self, n_agents, state_dim, action_dim, **kwargs):
        self.n_agents = n_agents
        
        # Create individual PPO agents
        self.agents = [
            PPOModel(state_dim, action_dim, **kwargs)
            for _ in range(n_agents)
        ]
        
        # Shared parameters (optional)
        self.shared_policy = kwargs.get('shared_policy', False)
        
        if self.shared_policy:
            # Share networks among agents
            for i in range(1, n_agents):
                self.agents[i].policy_net = self.agents[0].policy_net
                self.agents[i].optimizer = self.agents[0].optimizer
    
    def get_actions(self, states, deterministic=False):
        """Get actions for all agents"""
        actions = []
        logprobs = []
        values = []
        
        for i in range(self.n_agents):
            action, logprob, value = self.agents[i].get_action(states[i], deterministic)
            actions.append(action)
            logprobs.append(logprob)
            values.append(value)
        
        return np.array(actions), np.array(logprobs), np.array(values)
    
    def update(self, batch, n_epochs=4, batch_size=64):
        """Update all agents"""
        all_stats = []
        
        for i in range(self.n_agents):
            agent_batch = {
                'states': batch['states'][:, i, :],
                'actions': batch['actions'][:, i],
                'logprobs': batch['logprobs'][:, i],
                'rewards': batch['rewards'][:, i],
                'dones': batch['dones'][:, i],
                'values': batch['values'][:, i]
            }
            
            stats = self.agents[i].update(agent_batch, n_epochs, batch_size)
            all_stats.append(stats)
        
        # Average statistics
        avg_stats = {}
        for key in all_stats[0].keys():
            avg_stats[key] = np.mean([stats[key] for stats in all_stats])
        
        return avg_stats
    
    def save(self, path):
        """Save all agents"""
        checkpoint = {}
        
        for i, agent in enumerate(self.agents):
            agent_path = f"{path}_agent_{i}.pt"
            agent.save(agent_path)
            checkpoint[f'agent_{i}_path'] = agent_path
        
        # Save meta information
        meta = {
            'n_agents': self.n_agents,
            'shared_policy': self.shared_policy
        }
        
        torch.save(meta, f"{path}_meta.pt")
        print(f"Multi-agent PPO saved to {path}")