"""
Training script for Disaster Response AI - Week 4
Main training orchestration with support for multiple ML frameworks
"""
import os
import sys
import yaml
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    print("⚠️  Gymnasium not available")

try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️  Stable-Baselines3 not available")

try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPO as RllibPPO
    from ray.rllib.algorithms.a2c import A2C as RllibA2C
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("⚠️  Ray/RLlib not available")

from environments.simple_grid_env import SimpleGridEnv
from marl.pettingzoo_wrapper import DisasterResponsePettingZoo
from training.utils.model_loader import ModelLoader

class TrainingMetricsCallback(BaseCallback):
    """Custom callback for tracking training metrics"""
    
    def __init__(self, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Get reward from environment
        reward = self.locals.get('rewards', [0])[0] if 'rewards' in self.locals else 0
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Check if episode is done
        done = self.locals.get('dones', [False])[0] if 'dones' in self.locals else False
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Log every 10 episodes
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {len(self.episode_rewards)} - Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
        
        return True

class DisasterResponseTrainer:
    """Main training class for Disaster Response AI"""
    
    def __init__(self, config_path="training/configs/base_config.yaml"):
        self.config = self.load_config(config_path)
        self.model = None
        self.env = None
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'training_time': 0,
            'success_rate': 0
        }
        
    def load_config(self, config_path):
        """Load training configuration"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"⚠️  Config file {config_path} not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default training configuration"""
        return {
            'environment': {
                'type': 'simple_grid',
                'grid_size': 20,
                'num_civilians': 10,
                'num_disasters': 3,
                'max_steps': 1000
            },
            'training': {
                'algorithm': 'PPO',
                'total_timesteps': 100000,
                'learning_rate': 0.0003,
                'gamma': 0.99,
                'batch_size': 64,
                'n_epochs': 10
            },
            'agents': {
                'num_drones': 2,
                'num_ambulances': 2,
                'num_rescue_teams': 1
            }
        }
    
    def create_environment(self, render_mode=None):
        """Create training environment"""
        env_config = self.config['environment']
        
        if env_config['type'] == 'simple_grid':
            env = SimpleGridEnv(
                grid_size=env_config['grid_size'],
                num_civilians=env_config['num_civilians'],
                num_disasters=env_config['num_disasters'],
                max_steps=env_config['max_steps'],
                render_mode=render_mode
            )
            
            # Try to wrap with PettingZoo
            try:
                env = DisasterResponsePettingZoo(env)
                print("✅ Using PettingZoo multi-agent environment")
            except Exception as e:
                print(f"⚠️  PettingZoo wrapper failed: {e}")
                print("🔄 Using single-agent environment")
                
        else:
            raise ValueError(f"Unknown environment type: {env_config['type']}")
        
        return env
    
    def train_with_sb3(self):
        """Train using Stable-Baselines3"""
        if not SB3_AVAILABLE:
            print("❌ Stable-Baselines3 not available")
            return False
            
        print("🚀 Starting Stable-Baselines3 Training...")
        
        # Create environment
        self.env = self.create_environment()
        
        # For SB3, we need a gymnasium-compatible env
        # Create a simple wrapper if needed
        if not GYMNASIUM_AVAILABLE:
            print("❌ Gymnasium required for SB3 training")
            return False
            
        # Wrap environment for SB3
        try:
            sb3_env = self._wrap_for_sb3(self.env)
        except Exception as e:
            print(f"❌ Failed to wrap environment for SB3: {e}")
            return False
        
        # Initialize model
        algorithm = self.config['training']['algorithm'].upper()
        if algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                sb3_env,
                learning_rate=self.config['training']['learning_rate'],
                gamma=self.config['training']['gamma'],
                batch_size=self.config['training']['batch_size'],
                n_epochs=self.config['training']['n_epochs'],
                verbose=1,
                tensorboard_log="./logs/tensorboard/"
            )
        elif algorithm == 'A2C':
            self.model = A2C(
                'MlpPolicy',
                sb3_env,
                learning_rate=self.config['training']['learning_rate'],
                gamma=self.config['training']['gamma'],
                verbose=1,
                tensorboard_log="./logs/tensorboard/"
            )
        elif algorithm == 'DQN':
            self.model = DQN(
                'MlpPolicy',
                sb3_env,
                learning_rate=self.config['training']['learning_rate'],
                gamma=self.config['training']['gamma'],
                verbose=1,
                tensorboard_log="./logs/tensorboard/"
            )
        else:
            print(f"❌ Unsupported algorithm: {algorithm}")
            return False
        
        # Create callbacks
        callback = TrainingMetricsCallback()
        
        # Start training
        start_time = datetime.now()
        self.model.learn(
            total_timesteps=self.config['training']['total_timesteps'],
            callback=callback,
            tb_log_name=f"{algorithm}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        )
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save metrics
        self.metrics['episode_rewards'] = callback.episode_rewards
        self.metrics['episode_lengths'] = callback.episode_lengths
        self.metrics['training_time'] = training_time
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        return True
    
    def _wrap_for_sb3(self, env):
        """Wrap environment for Stable-Baselines3"""
        # Simple wrapper to make environment compatible with SB3
        class SB3Wrapper(gym.Env):
            def __init__(self, original_env):
                super().__init__()
                self.original_env = original_env
                self.action_space = gym.spaces.Discrete(6)  # Assuming 6 actions
                self.observation_space = gym.spaces.Box(
                    low=0, high=1, shape=(100,)  # Adjust shape as needed
                )
                
            def reset(self, **kwargs):
                obs, info = self.original_env.reset(**kwargs)
                return obs, info
                
            def step(self, action):
                return self.original_env.step(action)
                
            def render(self):
                return self.original_env.render()
                
            def close(self):
                return self.original_env.close()
        
        return SB3Wrapper(env)
    
    def train_with_custom(self):
        """Custom training implementation"""
        print("🚀 Starting Custom Training...")
        
        # Create environment
        self.env = self.create_environment()
        
        # Simple Q-learning implementation for demonstration
        state_size = 100  # Adjust based on your state representation
        action_size = 6   # Number of possible actions
        
        # Initialize Q-table
        q_table = np.zeros((state_size, action_size))
        
        # Training parameters
        learning_rate = 0.1
        discount_factor = 0.95
        epsilon = 1.0
        epsilon_decay = 0.995
        min_epsilon = 0.01
        episodes = 1000
        
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            state_idx = self._state_to_index(state)
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < self.config['environment']['max_steps']:
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(action_size)
                else:
                    action = np.argmax(q_table[state_idx])
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                next_state_idx = self._state_to_index(next_state)
                
                # Update Q-table
                old_value = q_table[state_idx, action]
                next_max = np.max(q_table[next_state_idx])
                
                new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
                q_table[state_idx, action] = new_value
                
                state_idx = next_state_idx
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            
            # Decay epsilon
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode} - Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
        
        self.metrics['episode_rewards'] = episode_rewards
        print("✅ Custom training completed")
        return True
    
    def _state_to_index(self, state):
        """Convert state to index for Q-table (simplified)"""
        if isinstance(state, (int, np.integer)):
            return state % 100
        elif isinstance(state, np.ndarray):
            return hash(state.tobytes()) % 100
        else:
            return hash(str(state)) % 100
    
    def evaluate_model(self, num_episodes=10):
        """Evaluate trained model"""
        if self.model is None:
            print("❌ No model trained yet")
            return
        
        print(f"🧪 Evaluating model over {num_episodes} episodes...")
        
        eval_env = self.create_environment(render_mode=None)
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = eval_env.reset()
            total_reward = 0
            done = False
            
            while not done:
                if SB3_AVAILABLE and isinstance(self.model, (PPO, A2C, DQN)):
                    action, _ = self.model.predict(state, deterministic=True)
                else:
                    # For custom models
                    action = self.model.act(state)
                
                state, reward, done, info = eval_env.step(action)
                total_reward += reward
            
            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")
        
        avg_reward = np.mean(episode_rewards)
        success_rate = np.mean([r > 0 for r in episode_rewards])
        
        print(f"📊 Evaluation Results:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Success Rate: {success_rate:.2%}")
        
        self.metrics['success_rate'] = success_rate
        return avg_reward, success_rate
    
    def save_model(self, filepath=None):
        """Save trained model"""
        if self.model is None:
            print("❌ No model to save")
            return False
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"models/disaster_response_{timestamp}"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            if SB3_AVAILABLE and isinstance(self.model, (PPO, A2C, DQN)):
                self.model.save(filepath)
            else:
                # Save custom models
                import pickle
                with open(filepath + '.pkl', 'wb') as f:
                    pickle.dump(self.model, f)
            
            print(f"✅ Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return False
    
    def plot_training_progress(self):
        """Plot training progress"""
        if not self.metrics['episode_rewards']:
            print("❌ No training data to plot")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Plot rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['episode_rewards'])
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # Plot moving average
        if len(self.metrics['episode_rewards']) > 100:
            moving_avg = np.convolve(self.metrics['episode_rewards'], 
                                   np.ones(100)/100, mode='valid')
            plt.plot(range(99, len(self.metrics['episode_rewards'])), moving_avg, 
                   'r-', linewidth=2, label='Moving Avg (100)')
            plt.legend()
        
        # Plot episode lengths if available
        if self.metrics['episode_lengths']:
            plt.subplot(1, 2, 2)
            plt.plot(self.metrics['episode_lengths'])
            plt.title('Episode Lengths')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    print("🚀 Disaster Response AI Training - Week 4")
    print("============================================================")
    
    # Check available frameworks
    print("🤖 Available ML Frameworks:")
    print(f"  • Gymnasium: {'✅' if GYMNASIUM_AVAILABLE else '❌'}")
    print(f"  • Stable-Baselines3: {'✅' if SB3_AVAILABLE else '❌'}")
    print(f"  • Ray/RLlib: {'✅' if RAY_AVAILABLE else '❌'}")
    
    # Initialize trainer
    trainer = DisasterResponseTrainer()
    
    # Training options
    print("\n🎯 Training Options:")
    print("1. Custom Training (Basic)")
    print("2. Stable-Baselines3 Training (Advanced)")
    print("3. Evaluate Existing Model")
    
    try:
        choice = input("🎯 Choose training method (1-3): ").strip()
        
        if choice == "1":
            success = trainer.train_with_custom()
        elif choice == "2":
            success = trainer.train_with_sb3()
        elif choice == "3":
            # Load existing model for evaluation
            model_loader = ModelLoader()
            model_path = input("Enter model path: ").strip()
            trainer.model = model_loader.load_model(model_path)
            if trainer.model:
                trainer.evaluate_model()
            return
        else:
            print("❌ Invalid choice")
            return
        
        if success:
            # Evaluate model
            trainer.evaluate_model()
            
            # Save model
            save_choice = input("💾 Save model? (y/n): ").strip().lower()
            if save_choice == 'y':
                trainer.save_model()
            
            # Plot results
            plot_choice = input("📊 Plot training progress? (y/n): ").strip().lower()
            if plot_choice == 'y':
                trainer.plot_training_progress()
            
            print("🎉 Week 4 Training Completed!")
            print("🤖 Your agents are now learning to coordinate!")
            print("🚀 Ready for Week 5: Advanced Features!")
    
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main()