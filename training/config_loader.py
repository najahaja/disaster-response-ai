"""
Configuration Loader for Multi-Agent RL System
Key Features: JSON-based config, validation, default value handling
"""

import json
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigLoader:
    """Configuration management class"""
    
    DEFAULT_CONFIG = {
        # Experiment
        'experiment_name': 'multi_agent_rl',
        'seed': 42,
        'use_gpu': True,
        
        # Environment
        'n_agents': 2,
        'train_grid_size': 10,
        'max_steps_per_episode': 100,
        'obs_type': 'grid',
        'use_real_map_for_testing': True,
        'map_file': 'data/maps/default_map.json',
        'use_physics': False,
        'sensor_noise': 0.05,
        
        # Algorithm
        'algorithm': 'ppo',  # 'ppo' or 'qmix'
        'gamma': 0.99,
        'learning_rate': 0.001,
        'hidden_dim': 128,
        'n_layers': 2,
        
        # PPO Specific
        'clip_epsilon': 0.2,
        'ppo_epochs': 4,
        'gae_lambda': 0.95,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        
        # QMIX Specific
        'mixing_hidden_dim': 32,
        'rnn_hidden_dim': 64,
        'use_rnn': True,
        'double_q': True,
        'dueling': False,
        'epsilon': 1.0,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.995,
        'target_update_freq': 200,
        'replay_buffer_size': 10000,
        
        # Training
        'total_episodes': 10000,
        'batch_size': 32,
        'save_freq': 100,
        'test_freq': 50,
        'print_freq': 10,
        'test_during_training': True,
        'target_success_rate': 0.8,
        
        # Reward Configuration
        'reward_config': {
            'goal_reward': 10.0,
            'collision_penalty': -5.0,
            'step_penalty': -0.1,
            'timeout_penalty': -2.0,
            'cooperation_bonus': 2.0
        },
        
        # Paths
        'checkpoint_dir': 'checkpoints',
        'model_dir': 'trained_models',
        'log_dir': 'logs',
        'result_dir': 'results',
        
        # Visualization
        'render_training': False,
        'render_testing': True,
        'save_videos': False,
        'fps': 30
    }
    
    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """
        Load configuration from JSON or YAML file
        Args:
            filepath: Path to configuration file
        Returns:
            Dictionary with configuration parameters
        """
        if not os.path.exists(filepath):
            print(f"Config file {filepath} not found. Using default configuration.")
            return ConfigLoader.DEFAULT_CONFIG.copy()
        
        # Determine file type
        file_extension = Path(filepath).suffix.lower()
        
        try:
            if file_extension == '.json':
                with open(filepath, 'r') as f:
                    user_config = json.load(f)
            elif file_extension in ['.yaml', '.yml']:
                with open(filepath, 'r') as f:
                    user_config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_extension}")
        except Exception as e:
            print(f"Error loading config file: {e}. Using default configuration.")
            return ConfigLoader.DEFAULT_CONFIG.copy()
        
        # Merge with defaults
        config = ConfigLoader.DEFAULT_CONFIG.copy()
        config = ConfigLoader._deep_merge(config, user_config)
        
        # Validate configuration
        ConfigLoader._validate_config(config)
        
        # Create directories
        ConfigLoader._create_directories(config)
        
        return config
    
    @staticmethod
    def _deep_merge(base: Dict, update: Dict) -> Dict:
        """Recursively merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = ConfigLoader._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    @staticmethod
    def _validate_config(config: Dict) -> None:
        """Validate configuration parameters"""
        # Algorithm validation
        if config['algorithm'] not in ['ppo', 'qmix']:
            raise ValueError(f"Invalid algorithm: {config['algorithm']}. Must be 'ppo' or 'qmix'")
        
        # Positive values validation
        positive_params = ['n_agents', 'train_grid_size', 'max_steps_per_episode', 
                          'total_episodes', 'batch_size', 'save_freq', 'test_freq']
        for param in positive_params:
            if config[param] <= 0:
                raise ValueError(f"{param} must be positive, got {config[param]}")
        
        # Probability validation
        prob_params = ['gamma', 'clip_epsilon', 'gae_lambda', 'epsilon', 
                       'epsilon_min', 'target_success_rate']
        for param in prob_params:
            if param in config and not (0 <= config[param] <= 1):
                raise ValueError(f"{param} must be between 0 and 1, got {config[param]}")
        
        # Learning rate validation
        if config['learning_rate'] <= 0:
            raise ValueError(f"learning_rate must be positive, got {config['learning_rate']}")
        
        # Path validation
        if config['use_real_map_for_testing'] and not os.path.exists(config['map_file']):
            print(f"Warning: Map file {config['map_file']} not found. RealMapEnv will use default.")
    
    @staticmethod
    def _create_directories(config: Dict) -> None:
        """Create necessary directories"""
        directories = [
            config['checkpoint_dir'],
            config['model_dir'],
            config['log_dir'],
            config['result_dir']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save(config: Dict, filepath: str) -> None:
        """
        Save configuration to file
        Args:
            config: Configuration dictionary
            filepath: Path to save configuration
        """
        file_extension = Path(filepath).suffix.lower()
        
        try:
            if file_extension == '.json':
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
            elif file_extension in ['.yaml', '.yml']:
                with open(filepath, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config file format: {file_extension}")
            
            print(f"Configuration saved to {filepath}")
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    @staticmethod
    def generate_template(filepath: str) -> None:
        """
        Generate a template configuration file
        Args:
            filepath: Path to save template
        """
        template = ConfigLoader.DEFAULT_CONFIG.copy()
        
        # Add comments/descriptions
        template['_comment'] = {
            'experiment_name': 'Name of the experiment for logging',
            'algorithm': 'Reinforcement learning algorithm (ppo or qmix)',
            'n_agents': 'Number of agents in the environment',
            'gamma': 'Discount factor for future rewards',
            'learning_rate': 'Learning rate for optimizer',
            'total_episodes': 'Total number of training episodes'
        }
        
        ConfigLoader.save(template, filepath)
        print(f"Template configuration generated at {filepath}")
    
    @staticmethod
    def update_from_args(config: Dict, args: Any) -> Dict:
        """
        Update configuration from command line arguments
        Args:
            config: Current configuration
            args: Command line arguments (from argparse)
        Returns:
            Updated configuration
        """
        arg_dict = vars(args)
        
        for key, value in arg_dict.items():
            if value is not None and key in config:
                config[key] = value
        
        return config