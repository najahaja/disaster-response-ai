"""
train.py - Complete Multi-Agent Reinforcement Learning Training System
Supports: PPO and QMIX algorithms, SimpleGridEnv for training, RealMapEnv for testing
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import json
import zipfile
from pathlib import Path

# Add custom imports
sys.path.append('./environments')
sys.path.append('./models')
sys.path.append('./utils')

# Custom imports
from simple_grid_env import SimpleGridEnv
from real_map_env import RealMapEnv
from ppo_model import PPOModel
from qmix_model import QMixModel
from replay_buffer import ReplayBuffer
from config_loader import ConfigLoader

class MultiAgentTrainer:
    """Main trainer class that handles both PPO and QMIX training"""
    
    def __init__(self, config_file='config.json'):
        # Load configuration
        self.config = ConfigLoader.load(config_file)
        
        # Set random seeds
        self._set_seeds()
        
        # Initialize environments
        self.train_env = None
        self.test_env = None
        
        # Initialize model
        self.model = None
        self.optimizer = None
        
        # Training state
        self.episode = 0
        self.best_score = -float('inf')
        self.scores_window = deque(maxlen=100)  # Last 100 scores
        self.losses = []
        
        # Create directories
        self._create_directories()
        
        # Initialize tensorboard
        self.writer = SummaryWriter(f"runs/{self.config['experiment_name']}")
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config['use_gpu'] else "cpu")
        print(f"Using device: {self.device}")
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        seed = self.config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _create_directories(self):
        """Create necessary directories"""
        Path("checkpoints").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("trained_models").mkdir(exist_ok=True)
    
    def initialize_environments(self):
        """Initialize training and testing environments"""
        print("Initializing environments...")
        
        # Training environment (SimpleGridEnv)
        train_config = {
            'grid_size': self.config['train_grid_size'],
            'n_agents': self.config['n_agents'],
            'n_goals': self.config['n_goals'],
            'max_steps': self.config['max_steps_per_episode'],
            'obs_type': self.config['obs_type'],
            'reward_config': self.config['reward_config']
        }
        
        self.train_env = SimpleGridEnv(**train_config)
        print(f"Training Environment: {self.train_env.grid_size}x{self.train_env.grid_size} grid")
        print(f"Number of agents: {self.train_env.n_agents}")
        print(f"Observation space: {self.train_env.observation_space}")
        print(f"Action space: {self.train_env.action_space}")
        
        # Testing environment (RealMapEnv) - only if we have map data
        if self.config['use_real_map_for_testing'] and self.config.get('map_file'):
            test_config = {
                'map_file': self.config['map_file'],
                'n_agents': self.config['n_agents'],
                'max_steps': self.config['max_steps_per_episode'],
                'use_physics': self.config['use_physics'],
                'noise_level': self.config.get('sensor_noise', 0.0)
            }
            self.test_env = RealMapEnv(**test_config)
            print(f"Testing Environment: RealMapEnv with {self.config['map_file']}")
        else:
            self.test_env = None
            print("Testing Environment: Using SimpleGridEnv for testing")
    
    def initialize_model(self):
        """Initialize the selected model (PPO or QMIX)"""
        algorithm = self.config['algorithm'].lower()
        
        if algorithm == 'ppo':
            self._initialize_ppo()
        elif algorithm == 'qmix':
            self._initialize_qmix()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Choose 'ppo' or 'qmix'")
        
        print(f"Initialized {algorithm.upper()} model")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Move model to device
        self.model.to(self.device)
    
    def _initialize_ppo(self):
        """Initialize PPO model"""
        # Get observation and action dimensions
        obs_dim = self.train_env.observation_space
        action_dim = self.train_env.action_space
        
        # PPO specific config
        ppo_config = {
            'state_dim': obs_dim,
            'action_dim': action_dim,
            'n_agents': self.config['n_agents'],
            'hidden_dim': self.config.get('hidden_dim', 128),
            'n_layers': self.config.get('n_layers', 2),
            'use_lstm': self.config.get('use_lstm', False),
            'gamma': self.config.get('gamma', 0.99),
            'gae_lambda': self.config.get('gae_lambda', 0.95),
            'clip_epsilon': self.config.get('clip_epsilon', 0.2),
            'value_coef': self.config.get('value_coef', 0.5),
            'entropy_coef': self.config.get('entropy_coef', 0.01)
        }
        
        self.model = PPOModel(**ppo_config)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            eps=1e-5
        )
    
    def _initialize_qmix(self):
        """Initialize QMIX model"""
        # Get observation and action dimensions
        obs_dim = self.train_env.observation_space
        action_dim = self.train_env.action_space
        
        # QMIX specific config
        qmix_config = {
            'n_agents': self.config['n_agents'],
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'hidden_dim': self.config.get('hidden_dim', 64),
            'mixing_hidden_dim': self.config.get('mixing_hidden_dim', 32),
            'rnn_hidden_dim': self.config.get('rnn_hidden_dim', 64),
            'use_rnn': self.config.get('use_rnn', True),
            'gamma': self.config.get('gamma', 0.99),
            'double_q': self.config.get('double_q', True),
            'dueling': self.config.get('dueling', False)
        }
        
        self.model = QMixModel(**qmix_config)
        
        # Optimizer
        self.optimizer = optim.RMSprop(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            alpha=0.99,
            eps=1e-5
        )
        
        # Replay buffer for QMIX
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.get('replay_buffer_size', 10000),
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_agents=self.config['n_agents']
        )
    
    def run_episode(self, env, training=True, render=False):
        """Run a single episode and return statistics"""
        algorithm = self.config['algorithm'].lower()
        
        if algorithm == 'ppo':
            return self._run_episode_ppo(env, training, render)
        elif algorithm == 'qmix':
            return self._run_episode_qmix(env, training, render)
    
    def _run_episode_ppo(self, env, training=True, render=False):
        """Run episode with PPO"""
        states = env.reset()
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        total_reward = 0
        steps = 0
        
        while steps < self.config['max_steps_per_episode']:
            if render:
                env.render()
                time.sleep(0.05)
            
            # Convert states to tensor
            states_tensor = torch.FloatTensor(states).to(self.device)
            
            # Get actions from policy
            with torch.set_grad_enabled(training):
                actions, log_probs, values = self.model(states_tensor)
            
            # Convert to numpy for environment
            if isinstance(actions, torch.Tensor):
                actions_np = actions.cpu().numpy()
            else:
                actions_np = actions
            
            # Take step in environment
            next_states, rewards, dones, info = env.step(actions_np)
            
            # Store data
            episode_data['states'].append(states)
            episode_data['actions'].append(actions_np)
            episode_data['rewards'].append(rewards)
            episode_data['values'].append(values.cpu().numpy())
            episode_data['log_probs'].append(log_probs.cpu().numpy())
            episode_data['dones'].append(dones)
            
            total_reward += np.sum(rewards)
            states = next_states
            steps += 1
            
            if all(dones) or steps >= self.config['max_steps_per_episode']:
                break
        
        episode_stats = {
            'total_reward': total_reward,
            'steps': steps,
            'success': info.get('success', False),
            'collisions': info.get('collisions', 0),
            'episode_data': episode_data
        }
        
        return episode_stats
    
    def _run_episode_qmix(self, env, training=True, render=False):
        """Run episode with QMIX"""
        states = env.reset()
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        total_reward = 0
        steps = 0
        
        # Initialize RNN hidden states if using RNN
        if self.model.use_rnn:
            hidden_states = self.model.init_hidden()
        else:
            hidden_states = None
        
        while steps < self.config['max_steps_per_episode']:
            if render:
                env.render()
                time.sleep(0.05)
            
            # Convert states to tensor
            states_tensor = torch.FloatTensor(states).to(self.device)
            
            # Get Q-values and actions
            with torch.set_grad_enabled(training):
                if self.model.use_rnn:
                    q_values, hidden_states = self.model(states_tensor, hidden_states)
                else:
                    q_values = self.model(states_tensor)
            
            # Epsilon-greedy exploration
            epsilon = self.config.get('epsilon', 0.1)
            if training and random.random() < epsilon:
                actions = [random.randint(0, env.action_space - 1) for _ in range(env.n_agents)]
            else:
                # Choose actions with highest Q-values
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
            
            # Take step in environment
            next_states, rewards, dones, info = env.step(actions)
            
            # Store data in replay buffer
            if training:
                self.replay_buffer.push(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=dones
                )
            
            # Store for episode data
            episode_data['states'].append(states)
            episode_data['actions'].append(actions)
            episode_data['rewards'].append(rewards)
            episode_data['next_states'].append(next_states)
            episode_data['dones'].append(dones)
            
            total_reward += np.sum(rewards)
            states = next_states
            steps += 1
            
            if all(dones) or steps >= self.config['max_steps_per_episode']:
                break
        
        episode_stats = {
            'total_reward': total_reward,
            'steps': steps,
            'success': info.get('success', False),
            'collisions': info.get('collisions', 0),
            'episode_data': episode_data
        }
        
        return episode_stats
    
    def update_model(self, episode_data):
        """Update model based on collected episode data"""
        algorithm = self.config['algorithm'].lower()
        
        if algorithm == 'ppo':
            return self._update_ppo(episode_data)
        elif algorithm == 'qmix':
            return self._update_qmix()
    
    def _update_ppo(self, episode_data):
        """Update PPO model"""
        # Calculate advantages
        rewards = np.array(episode_data['rewards'])
        values = np.array(episode_data['values'])
        dones = np.array(episode_data['dones'])
        
        # Compute returns and advantages
        returns, advantages = self.model.compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=self.config.get('gamma', 0.99),
            gae_lambda=self.config.get('gae_lambda', 0.95)
        )
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(episode_data['states'])).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(episode_data['actions'])).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(np.array(episode_data['log_probs'])).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # PPO update
        losses = self.model.update(
            states=states_tensor,
            actions=actions_tensor,
            old_log_probs=old_log_probs_tensor,
            returns=returns_tensor,
            advantages=advantages_tensor,
            optimizer=self.optimizer,
            clip_epsilon=self.config.get('clip_epsilon', 0.2),
            n_epochs=self.config.get('ppo_epochs', 4),
            batch_size=self.config.get('batch_size', 32)
        )
        
        return losses
    
    def _update_qmix(self):
        """Update QMIX model"""
        # Sample batch from replay buffer
        if len(self.replay_buffer) < self.config.get('batch_size', 32):
            return None
        
        batch = self.replay_buffer.sample(self.config.get('batch_size', 32))
        
        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # QMIX update
        loss = self.model.update(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            optimizer=self.optimizer,
            gamma=self.config.get('gamma', 0.99),
            target_update_freq=self.config.get('target_update_freq', 200)
        )
        
        return {'qmix_loss': loss}
    
    def save_checkpoint(self, score, episode, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': score,
            'config': self.config,
            'best_score': self.best_score
        }
        
        # Regular checkpoint
        checkpoint_path = f"checkpoints/checkpoint_ep{episode}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best model if improved
        if is_best:
            self.best_score = score
            best_path = "checkpoints/best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with score: {score:.2f}")
            
            # Also save as .zip file for deployment
            self._save_as_zip(episode, score)
    
    def _save_as_zip(self, episode, score):
        """Save model in .zip format for deployment"""
        zip_path = f"trained_models/{self.config['experiment_name']}_ep{episode}_score{score:.2f}.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Save model architecture and weights
            model_info = {
                'algorithm': self.config['algorithm'],
                'n_agents': self.config['n_agents'],
                'obs_dim': self.train_env.observation_space,
                'action_dim': self.train_env.action_space,
                'episode': episode,
                'score': score,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }
            
            # Save model info as JSON
            with zipf.open('model_info.json', 'w') as f:
                f.write(json.dumps(model_info, indent=2).encode())
            
            # Save model weights
            model_weights_path = f"temp_model_weights.pt"
            torch.save(self.model.state_dict(), model_weights_path)
            zipf.write(model_weights_path, 'model_weights.pt')
            os.remove(model_weights_path)
            
            # Save configuration
            config_str = json.dumps(self.config, indent=2)
            zipf.writestr('config.json', config_str)
            
            # Save training stats if available
            if hasattr(self, 'scores_window'):
                stats = {
                    'mean_score': np.mean(self.scores_window) if self.scores_window else 0,
                    'max_score': max(self.scores_window) if self.scores_window else 0,
                    'losses': self.losses[-100:] if self.losses else []
                }
                zipf.writestr('training_stats.json', json.dumps(stats, indent=2))
        
        print(f"Model saved as ZIP: {zip_path}")
        return zip_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episode = checkpoint['episode']
            self.best_score = checkpoint.get('best_score', -float('inf'))
            print(f"Loaded checkpoint from episode {self.episode}")
            return True
        return False
    
    def test_model(self, n_episodes=10, render=False):
        """Test the model on testing environment"""
        if self.test_env is None:
            print("No testing environment available. Using training environment.")
            test_env = self.train_env
        else:
            test_env = self.test_env
        
        print(f"\nTesting model on {n_episodes} episodes...")
        
        test_scores = []
        test_successes = []
        test_steps = []
        
        for ep in range(n_episodes):
            # Set model to evaluation mode
            self.model.eval()
            
            # Run test episode
            with torch.no_grad():
                episode_stats = self.run_episode(test_env, training=False, render=render)
            
            test_scores.append(episode_stats['total_reward'])
            test_successes.append(episode_stats['success'])
            test_steps.append(episode_stats['steps'])
            
            print(f"Test Episode {ep+1}: Score={episode_stats['total_reward']:.2f}, "
                  f"Steps={episode_stats['steps']}, Success={episode_stats['success']}")
        
        # Set back to training mode
        self.model.train()
        
        test_results = {
            'mean_score': np.mean(test_scores),
            'std_score': np.std(test_scores),
            'success_rate': np.mean(test_successes),
            'mean_steps': np.mean(test_steps),
            'scores': test_scores
        }
        
        print(f"\nTest Results:")
        print(f"  Mean Score: {test_results['mean_score']:.2f} ± {test_results['std_score']:.2f}")
        print(f"  Success Rate: {test_results['success_rate']:.2%}")
        print(f"  Mean Steps: {test_results['mean_steps']:.1f}")
        
        # Log to tensorboard
        self.writer.add_scalar('Test/Mean_Score', test_results['mean_score'], self.episode)
        self.writer.add_scalar('Test/Success_Rate', test_results['success_rate'], self.episode)
        
        return test_results
    
    def train(self, n_episodes=None):
        """Main training loop"""
        if n_episodes is None:
            n_episodes = self.config['total_episodes']
        
        print(f"\nStarting training for {n_episodes} episodes...")
        print(f"Algorithm: {self.config['algorithm'].upper()}")
        print(f"Environment: {self.train_env.__class__.__name__}")
        
        # Training loop
        for ep in range(self.episode + 1, self.episode + n_episodes + 1):
            self.episode = ep
            
            # Run training episode
            episode_start_time = time.time()
            episode_stats = self.run_episode(self.train_env, training=True, render=False)
            episode_time = time.time() - episode_start_time
            
            # Update model
            update_start_time = time.time()
            losses = self.update_model(episode_stats['episode_data'])
            update_time = time.time() - update_start_time
            
            # Store score
            score = episode_stats['total_reward']
            self.scores_window.append(score)
            if losses:
                self.losses.append(losses)
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Score', score, ep)
            self.writer.add_scalar('Train/Mean_Score_100', np.mean(self.scores_window), ep)
            self.writer.add_scalar('Train/Steps', episode_stats['steps'], ep)
            if losses:
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f'Train/{loss_name}', loss_value, ep)
            
            # Print progress
            if ep % self.config['print_freq'] == 0:
                print(f"\nEpisode {ep}/{self.episode + n_episodes}")
                print(f"  Score: {score:.2f}")
                print(f"  Avg Score (last 100): {np.mean(self.scores_window):.2f}")
                print(f"  Steps: {episode_stats['steps']}")
                print(f"  Success: {episode_stats['success']}")
                print(f"  Collisions: {episode_stats.get('collisions', 0)}")
                print(f"  Episode Time: {episode_time:.2f}s")
                print(f"  Update Time: {update_time:.2f}s")
                if losses:
                    loss_str = ', '.join([f"{k}: {v:.4f}" for k, v in losses.items()])
                    print(f"  Losses: {loss_str}")
            
            # Save checkpoint
            if ep % self.config['save_freq'] == 0:
                is_best = score > self.best_score
                self.save_checkpoint(score, ep, is_best)
            
            # Test periodically
            if ep % self.config['test_freq'] == 0 and self.config.get('test_during_training', True):
                test_results = self.test_model(n_episodes=5, render=False)
                
                # Early stopping if test performance is good
                if test_results['success_rate'] >= self.config.get('target_success_rate', 0.95):
                    print(f"\nEarly stopping: Achieved target success rate of {test_results['success_rate']:.2%}")
                    break
            
            # Learning rate decay
            if ep % self.config.get('lr_decay_freq', 1000) == 0:
                self._decay_learning_rate()
        
        print("\nTraining completed!")
        
        # Final test
        final_test_results = self.test_model(n_episodes=20, render=True)
        
        # Save final model
        final_zip_path = self._save_as_zip(self.episode, final_test_results['mean_score'])
        
        # Close tensorboard writer
        self.writer.close()
        
        return final_zip_path, final_test_results
    
    def _decay_learning_rate(self):
        """Decay learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.config.get('lr_decay', 0.99)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.train_env:
            self.train_env.close()
        if self.test_env:
            self.test_env.close()
        self.writer.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multi-Agent RL Training System')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--test-only', action='store_true', help='Only test, no training')
    parser.add_argument('--render', action='store_true', help='Render during testing')
    parser.add_argument('--episodes', type=int, help='Number of episodes to train')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MultiAgentTrainer(config_file=args.config)
    
    # Initialize environments and model
    trainer.initialize_environments()
    trainer.initialize_model()
    
    # Resume from checkpoint if specified
    if args.resume:
        if trainer.load_checkpoint(args.resume):
            print(f"Resumed training from checkpoint")
        else:
            print(f"Could not load checkpoint: {args.resume}")
    
    # Test only mode
    if args.test_only:
        test_results = trainer.test_model(n_episodes=20, render=args.render)
        
        # Save test results
        with open('test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        trainer.cleanup()
        return
    
    # Train
    final_model_path, final_results = trainer.train(n_episodes=args.episodes)
    
    # Save final results
    results = {
        'final_model': final_model_path,
        'test_results': final_results,
        'best_score': trainer.best_score,
        'total_episodes': trainer.episode
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Final model saved as: {final_model_path}")
    print(f"Best score achieved: {trainer.best_score:.2f}")
    
    # Cleanup
    trainer.cleanup()

if __name__ == "__main__":
    main()