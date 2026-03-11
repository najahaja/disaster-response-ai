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
# Add project root and subdirectories to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

sys.path.append(os.path.join(project_root, 'environments'))
sys.path.append(os.path.join(project_root, 'training'))
sys.path.append(os.path.join(project_root, 'utils'))

# Custom imports
from environments.simple_grid_env import SimpleGridEnv
from environments.real_map_env import RealMapEnv
from training.ppo_model import PPOModel
from training.qmix_model import QMIXModel
from training.replay_buffer import ReplayBuffer
from training.config_loader import ConfigLoader
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
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.get('use_gpu', True) else "cpu")
        print(f"Using device: {self.device}")
        
        # For tracking training time
        self.start_time = time.time()
    
    def _create_default_config(self):
        """Create default config"""
        return {
            'experiment_name': 'multi_agent_rl',
            'algorithm': 'ppo',
            'n_agents': 2,
            'train_grid_size': 10,
            'max_steps_per_episode': 100,
            'total_episodes': 200,  # Reduced for testing
            'learning_rate': 3e-5,
            'use_gpu': True,
            'save_freq': 1000,  # Reduced for testing
            'test_freq': 2000,  # Reduced for testing
            'print_freq': 500,
            'seed': 42,
            'gamma': 0.99,
            'hidden_dim': 64,  # Reduced for testing
            'n_layers': 2,
            'batch_size': 16,  # Reduced for testing
            'ppo_epochs': 2,  # Reduced for testing
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.08,
            'test_during_training': False  # Disabled for testing
        }
    
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
        # Now supports configurable agents and civilians
        grid_size = self.config.get('train_grid_size', 36)
        config_path = self.config.get('env_config_path', 'config.yaml')
        
        # Agent configuration
        n_drones = self.config.get('n_drones', 1)
        n_ambulances = self.config.get('n_ambulances', 1)
        n_rescue_teams = self.config.get('n_rescue_teams', 1)
        
        # Civilian configuration
        spawn_civilians = self.config.get('spawn_civilians', True)
        n_civilians = self.config.get('n_civilians', None)  # None = use config.yaml default
        
        self.train_env = SimpleGridEnv(
            grid_size=grid_size, 
            config_path=config_path,
            n_drones=n_drones,
            n_ambulances=n_ambulances,
            n_rescue_teams=n_rescue_teams,
            spawn_civilians=spawn_civilians,
            n_civilians=n_civilians
        )
        
        print(f"Training Environment: {self.train_env.grid_size}x{self.train_env.grid_size} grid")
        print(f"  Agents: {n_drones} drone(s), {n_ambulances} ambulance(s), {n_rescue_teams} rescue team(s)")
        print(f"  Civilians: {'Enabled' if spawn_civilians else 'Disabled'}" + 
              (f" ({n_civilians} civilians)" if n_civilians else ""))
        print(f"Observation space: {self.train_env.observation_space}")
        print(f"Action space: {self.train_env.action_space}")
        
        # Testing environment (RealMapEnv) - only if we have map data
        if self.config.get('use_real_map_for_testing', False) and self.config.get('map_file'):
            test_config = {
                'map_file': self.config['map_file'],
                'n_agents': self.config.get('n_agents', 3),
                'max_steps': self.config.get('max_steps_per_episode', 1000),
                'use_physics': self.config.get('use_physics', False),
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
        print(f"Total parameters: {sum(p.numel() for p in self.model.policy_net.parameters()):,}")
        
        # Move model to device
        self.model.policy_net.to(self.device)
    
    def _initialize_ppo(self):
        """Initialize PPO model"""
        # Get observation and action dimensions from Gym spaces
        # observation_space is a Box, action_space is Discrete
        obs_space = self.train_env.observation_space
        action_space = self.train_env.action_space
        
        # Extract dimensions properly
        if hasattr(obs_space, 'shape'):
            # For Box spaces, flatten the observation
            obs_dim = int(np.prod(obs_space.shape))
        else:
            obs_dim = obs_space.n
            
        if hasattr(action_space, 'n'):
            # For Discrete spaces
            action_dim = action_space.n
        else:
            action_dim = action_space.shape[0]
        
        # PPO specific config - only include parameters that PPOModel accepts
        ppo_config = {
            'state_dim': obs_dim,
            'action_dim': action_dim,
            'n_agents': self.config.get('n_agents', 1),
            'hidden_dim': self.config.get('hidden_dim', 128),
            'n_layers': self.config.get('n_layers', 2),
            'continuous': False,  # Discrete action space
            'gamma': self.config.get('gamma', 0.99),
            'gae_lambda': self.config.get('gae_lambda', 0.95),
            'clip_epsilon': self.config.get('clip_epsilon', 0.2),
            'value_coef': self.config.get('value_coef', 0.5),
            'entropy_coef': self.config.get('entropy_coef', 0.01),
            'max_grad_norm': self.config.get('max_grad_norm', 0.5)
        }
        
        self.model = PPOModel(**ppo_config)
        
        # PPOModel creates its own optimizer, but we'll override it with our learning rate
        self.optimizer = optim.Adam(
            self.model.policy_net.parameters(),
            lr=self.config.get('learning_rate', 0.0003),
            eps=1e-5
        )
        self.model.optimizer = self.optimizer
    
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
        # Gymnasium API: reset() returns (observation, info)
        states, info = env.reset()
        
        # Flatten the observation if it's a multi-dimensional array
        states_flat = states.flatten()
        
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
        done = False
        
        while steps < self.config.get('max_steps_per_episode', 1000) and not done:
            if render:
                env.render()
                time.sleep(0.05)
            
            # Convert states to tensor (flatten if needed)
            states_tensor = torch.FloatTensor(states_flat).unsqueeze(0).to(self.device)
            
            # Get actions from policy
            with torch.set_grad_enabled(training):
                actions, log_probs, values = self.model.policy_net.get_action(states_tensor)
            
            # Convert to numpy for environment (single action for single agent)
            if isinstance(actions, torch.Tensor):
                action = actions.cpu().numpy().item() if actions.numel() == 1 else actions.cpu().numpy()[0]
            else:
                action = actions
            
            # Take step in environment
            # Gymnasium API: step() returns (observation, reward, terminated, truncated, info)
            next_states, reward, terminated, truncated, info = env.step(action)
            next_states_flat = next_states.flatten()
            
            # Combine terminated and truncated into done
            done = terminated or truncated
            
            # Store data
            episode_data['states'].append(states_flat)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['values'].append(values.cpu().numpy())
            episode_data['log_probs'].append(log_probs.cpu().numpy())
            episode_data['dones'].append(done)
            
            total_reward += reward
            states_flat = next_states_flat
            steps += 1
        
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
        # Gymnasium API: reset() returns (observation, info)
        states, info = env.reset()
        states_flat = states.flatten()
        
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        total_reward = 0
        steps = 0
        done = False
        
        # Initialize RNN hidden states if using RNN
        if hasattr(self.model, 'use_rnn') and self.model.use_rnn:
            hidden_states = self.model.init_hidden()
        else:
            hidden_states = None
        
        while steps < self.config.get('max_steps_per_episode', 1000) and not done:
            if render:
                env.render()
                time.sleep(0.05)
            
            # Convert states to tensor
            states_tensor = torch.FloatTensor(states_flat).unsqueeze(0).to(self.device)
            
            # Get Q-values and actions
            with torch.set_grad_enabled(training):
                if hidden_states is not None:
                    q_values, hidden_states = self.model(states_tensor, hidden_states)
                else:
                    q_values = self.model(states_tensor)
            
            # Epsilon-greedy exploration
            epsilon = self.config.get('epsilon', 0.1)
            if training and random.random() < epsilon:
                # For single agent, get action space size
                if hasattr(env.action_space, 'n'):
                    action = random.randint(0, env.action_space.n - 1)
                else:
                    action = random.randint(0, 5)  # Default to 6 actions
            else:
                # Choose action with highest Q-value
                action = torch.argmax(q_values).cpu().numpy().item()
            
            # Take step in environment
            # Gymnasium API: step() returns (observation, reward, terminated, truncated, info)
            next_states, reward, terminated, truncated, info = env.step(action)
            next_states_flat = next_states.flatten()
            
            # Combine terminated and truncated into done
            done = terminated or truncated
            
            # Store data in replay buffer
            if training and hasattr(self, 'replay_buffer'):
                self.replay_buffer.push(
                    states=states_flat,
                    actions=action,
                    rewards=reward,
                    next_states=next_states_flat,
                    dones=done
                )
            
            # Store for episode data
            episode_data['states'].append(states_flat)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['next_states'].append(next_states_flat)
            episode_data['dones'].append(done)
            
            total_reward += reward
            states_flat = next_states_flat
            steps += 1
        
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
        # PPOModel.update expects a batch dict with specific keys
        # and will compute GAE internally
        
        # Prepare batch dictionary
        batch = {
            'states': np.array(episode_data['states']),
            'actions': np.array(episode_data['actions']),
            'logprobs': np.array(episode_data['log_probs']).squeeze(),
            'rewards': np.array(episode_data['rewards']),
            'dones': np.array(episode_data['dones']),
            'values': np.array(episode_data['values']).squeeze()
        }
        
        # PPO update (it will handle tensor conversion and GAE computation internally)
        losses = self.model.update(
            batch=batch,
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
            'model_state_dict': self.model.policy_net.state_dict(),
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
        
        try:
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Save model info
                model_info = {
                    'algorithm': self.config['algorithm'],
                    'n_agents': self.config.get('n_agents', 1),
                    'episode': episode,
                    'score': score,
                    'timestamp': time.strftime("%Y%m%d-%H%M%S"),
                    'device': str(self.device)
                }
                
                zipf.writestr('model_info.json', json.dumps(model_info, indent=2))
                
                # Save model weights
                model_path = f"temp_model_{episode}.pt"
                if hasattr(self.model, 'state_dict'):
                    torch.save(self.model.state_dict(), model_path)
                elif hasattr(self.model, 'policy_net'):
                    torch.save(self.model.policy_net.state_dict(), model_path)
                
                zipf.write(model_path, 'model_weights.pt')
                os.remove(model_path)
                
                # Save configuration
                config_str = json.dumps(self.config, indent=2)
                zipf.writestr('config.json', config_str)
                
                print(f"✓ Model saved as ZIP: {zip_path}")
                return zip_path
                
        except Exception as e:
            print(f"⚠ Failed to save ZIP: {e}")
            return None
    
    def train(self, n_episodes=None):
        """Main training loop - SIMPLIFIED FOR TESTING"""
        if n_episodes is None:
            n_episodes = self.config.get('total_episodes', 10000)  # Reduced for testing
        
        print("\n" + "="*60)
        print(f"STARTING TRAINING (TEST MODE)")
        print("="*60)
        print(f"Algorithm: {self.config['algorithm'].upper()}")
        print(f"Total episodes: {n_episodes}")
        print(f"Device: {self.device}")
        print(f"Environment: {self.train_env.__class__.__name__}")
        print("="*60 + "\n")
        
        # Training loop
        start_episode = self.episode
        total_start_time = time.time()
        
        try:
            for ep in range(start_episode + 1, start_episode + n_episodes + 1):
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
                        if loss_value is not None:
                            self.writer.add_scalar(f'Train/{loss_name}', loss_value, ep)
                
                # Print progress
                if ep % self.config.get('print_freq', 5) == 0 or ep == start_episode + n_episodes:
                    elapsed_time = time.time() - total_start_time
                    episodes_done = ep - start_episode
                    episodes_per_sec = episodes_done / elapsed_time if elapsed_time > 0 else 0
                    
                    print(f"\nEpisode {ep}/{start_episode + n_episodes}")
                    print(f"  Score: {score:.2f}")
                    print(f"  Avg Score (last 100): {np.mean(self.scores_window):.2f}")
                    print(f"  Steps: {episode_stats['steps']}")
                    print(f"  Episode Time: {episode_time:.2f}s")
                    print(f"  Update Time: {update_time:.2f}s")
                    print(f"  Speed: {episodes_per_sec:.1f} episodes/sec")
                    
                    if losses:
                        loss_str = ', '.join([f"{k}: {v:.4f}" for k, v in losses.items() if v is not None])
                        if loss_str:
                            print(f"  Losses: {loss_str}")
                
                # Save checkpoint
                if ep % self.config.get('save_freq', 10) == 0 or ep == start_episode + n_episodes:
                    is_best = score > self.best_score
                    self.save_checkpoint(score, ep, is_best)
                if ep % 20000 == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"📉 Milestone reached! Learning rate decayed to: {current_lr}")

                elapsed_hours = (time.time() - self.start_time) / 3600
                if elapsed_hours > 10.5:  # If we've been running for 10.5 hours
                    print("⏳ Time limit approaching. Saving final checkpoint and exiting...")
                    self.save_checkpoint(score, ep, is_best=True)
                    return {"status": "timed_out", "last_ep": ep}
            # Training completed
            total_time = time.time() - total_start_time
            print(f"\n" + "="*60)
            print(f"TRAINING COMPLETED")
            print("="*60)
            print(f"Total episodes: {self.episode}")
            print(f"Training time: {total_time:.1f} seconds")
            print(f"Best score: {self.best_score:.2f}")
            print(f"Average score (last 100): {np.mean(self.scores_window):.2f}")
            print("="*60)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving checkpoint...")
            self.save_checkpoint(score, ep, is_best=False)
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.policy_net.load_state_dict(checkpoint['model_state_dict'])
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
            self.model.policy_net.eval()
            
            # Run test episode
            with torch.no_grad():
                episode_stats = self.run_episode(test_env, training=False, render=render)
            
            test_scores.append(episode_stats['total_reward'])
            test_successes.append(episode_stats['success'])
            test_steps.append(episode_stats['steps'])
            
            print(f"Test Episode {ep+1}: Score={episode_stats['total_reward']:.2f}, "
                  f"Steps={episode_stats['steps']}, Success={episode_stats['success']}")
        
        # Set back to training mode
        self.model.policy_net.train()
        
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