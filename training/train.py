import os
import yaml
import numpy as np
from typing import Dict, List, Optional
import pygame

# Try to import RL frameworks
try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️  Stable-Baselines3 not available. Using custom training loop.")

try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPO as RllibPPO
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("⚠️  Ray/RLlib not available.")

from environments.simple_grid_env import SimpleGridEnv
from .utils.training_visualizer import TrainingVisualizer

class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.rescue_rates = []
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        # Log training metrics
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info['r'])
            self.episode_lengths.append(ep_info['l'])

class TrainingManager:
    """
    Main training manager for the disaster response environment
    """
    
    def __init__(self, config_path="training/configs/base_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize environment - but handle potential PettingZoo issues
        try:
            from marl.pettingzoo_wrapper import DisasterResponseEnv
            self.env = DisasterResponseEnv()
            self.pettingzoo_available = True
            print("✅ Using PettingZoo environment")
        except Exception as e:
            print(f"⚠️  PettingZoo wrapper not available: {e}")
            print("🔄 Falling back to basic environment...")
            from environments.simple_grid_env import SimpleGridEnv
            self.env = SimpleGridEnv()
            self.pettingzoo_available = False
        
        self.visualizer = TrainingVisualizer()
        
        # Training state
        self.model = None
        self.training_data = {
            'episode_rewards': [],
            'episode_lengths': [],
            'rescue_rates': [],
            'collaboration_scores': []
        }
    
    def train_with_sb3(self):
        """Train using Stable-Baselines3"""
        if not SB3_AVAILABLE:
            print("❌ Stable-Baselines3 not available")
            return
        
        print("🚀 Starting training with Stable-Baselines3...")
        
        # For multi-agent with SB3, we need to use a wrapper
        # This is a simplified approach - in practice, you'd need a multi-agent wrapper
        
        try:
            # Create a single-agent wrapper for demonstration
            from stable_baselines3.common.env_checker import check_env
            check_env(self.env)
            
            # Initialize PPO model
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config['training']['learning_rate'],
                n_steps=self.config['training']['n_steps'],
                batch_size=self.config['training']['batch_size'],
                n_epochs=self.config['training']['n_epochs'],
                gamma=self.config['training']['gamma'],
                gae_lambda=self.config['training']['gae_lambda'],
                clip_range=self.config['training']['clip_range'],
                verbose=1,
                tensorboard_log=self.config['logging']['tensorboard_log']
            )
            
            # Train the model
            callback = TrainingCallback()
            self.model.learn(
                total_timesteps=self.config['training']['total_timesteps'],
                callback=callback
            )
            
            # Store training data
            self.training_data['episode_rewards'] = callback.episode_rewards
            self.training_data['episode_lengths'] = callback.episode_lengths
            
            print("✅ Training completed!")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
    
    def train_custom(self, num_episodes=1000):
        """Custom training loop for multi-agent RL"""
        print("🚀 Starting custom multi-agent training...")
        
        # Simple training implementation for demonstration
        for episode in range(num_episodes):
            observation = self.env.reset()
            episode_reward = 0
            steps = 0
            
            # ✅ FIXED: Handle both PettingZoo and basic environments
            if self.pettingzoo_available:
                # PettingZoo environment
                while not self.env.env.is_done():
                    # Simple random policy for demonstration
                    actions = {}
                    for agent_id in self.env.agents:
                        action = np.random.randint(0, 6)  # Random action
                        actions[agent_id] = action
                    
                    # Take step with all actions
                    for agent_id, action in actions.items():
                        self.env.step(action)
                    
                    steps += 1
            else:
                # Basic environment
                while not self.env.is_done():
                    # Simple random policy for demonstration
                    actions = {}
                    for agent_id in self.env.agents:
                        action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'])
                        actions[agent_id] = action
                    
                    # Take step with all actions
                    observation, rewards, done, info = self.env.step(actions)
                    episode_reward += sum(rewards.values())
                    steps += 1
                    
                    if done:
                        break
            
            # Calculate episode metrics
            total_civilians = len(self.env.civilians)
            rescued_civilians = sum(1 for c in self.env.civilians if c['rescued'])
            rescue_rate = rescued_civilians / max(total_civilians, 1)
            
            # Store metrics
            self.training_data['episode_rewards'].append(episode_reward)
            self.training_data['episode_lengths'].append(steps)
            self.training_data['rescue_rates'].append(rescue_rate)
            
            if episode % 10 == 0:  # Print more frequently for testing
                print(f"📊 Episode {episode}: Reward={episode_reward:.1f}, "
                      f"Steps={steps}, Rescue Rate={rescue_rate:.2f}")
        
        print("✅ Custom training completed!")
    
    def evaluate(self, num_episodes=10):
        """Evaluate the trained model"""
        if self.model is None:
            print("❌ No model trained yet")
            return
        
        print("🧪 Evaluating model...")
        
        episode_rewards = []
        rescue_rates = []
        
        for episode in range(num_episodes):
            observation = self.env.reset()
            episode_reward = 0
            done = False
            
            if self.pettingzoo_available:
                # PettingZoo evaluation
                while not self.env.env.is_done():
                    if SB3_AVAILABLE and isinstance(self.model, (PPO, DQN, A2C)):
                        action, _ = self.model.predict(observation, deterministic=True)
                    else:
                        action = self.model.get_action(observation)  # Custom models
                    
                    observation, reward, done, info = self.env.step(action)
                    episode_reward += reward
            else:
                # Basic environment evaluation
                while not self.env.is_done():
                    # For basic env, use random actions for now
                    actions = {}
                    for agent_id in self.env.agents:
                        action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'])
                        actions[agent_id] = action
                    
                    observation, rewards, done, info = self.env.step(actions)
                    episode_reward += sum(rewards.values())
                    
                    if done:
                        break
            
            # Calculate rescue rate
            total_civilians = len(self.env.civilians)
            rescued_civilians = sum(1 for c in self.env.civilians if c['rescued'])
            rescue_rate = rescued_civilians / max(total_civilians, 1)
            
            episode_rewards.append(episode_reward)
            rescue_rates.append(rescue_rate)
        
        avg_reward = np.mean(episode_rewards)
        avg_rescue_rate = np.mean(rescue_rates)
        
        print(f"📊 Evaluation Results:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Rescue Rate: {avg_rescue_rate:.2f}")
        
        return avg_reward, avg_rescue_rate
    
    def visualize_training(self):
        """Visualize training results"""
        self.visualizer.plot_training_curves(self.training_data)
    
    def save_model(self, model_name):
        """Save the trained model"""
        from .utils.model_loader import ModelLoader
        
        loader = ModelLoader()
        metadata = {
            'training_config': self.config,
            'training_data': self.training_data
        }
        
        loader.save_model(self.model, model_name, metadata)
    
    def load_model(self, model_name):
        """Load a trained model"""
        from .utils.model_loader import ModelLoader
        
        loader = ModelLoader()
        self.model, metadata = loader.load_model(model_name)
        
        if 'training_data' in metadata:
            self.training_data = metadata['training_data']
        
        print("✅ Model loaded successfully!")