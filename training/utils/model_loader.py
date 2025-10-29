"""
Model loading utilities for Disaster Response AI - Week 4
Handles loading of trained models from different frameworks
"""
import os
import pickle
import numpy as np
from datetime import datetime

try:
    from stable_baselines3 import PPO, A2C, DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class ModelLoader:
    """Utility class for loading and managing trained models"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.loaded_models = {}
        
    def list_available_models(self):
        """List all available trained models"""
        if not os.path.exists(self.models_dir):
            print(f"❌ Models directory '{self.models_dir}' not found")
            return []
        
        models = []
        for filename in os.listdir(self.models_dir):
            if filename.endswith(('.zip', '.pkl', '.pt', '.pth')):
                model_path = os.path.join(self.models_dir, filename)
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                models.append({
                    'name': filename,
                    'path': model_path,
                    'size_mb': round(file_size, 2),
                    'modified': datetime.fromtimestamp(os.path.getmtime(model_path))
                })
        
        # Sort by modification time (newest first)
        models.sort(key=lambda x: x['modified'], reverse=True)
        return models
    
    def load_model(self, model_path, model_type="auto"):
        """
        Load a trained model
        
        Args:
            model_path: Path to model file
            model_type: Type of model ('sb3', 'custom', 'auto')
        """
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None
        
        try:
            # Auto-detect model type
            if model_type == "auto":
                if model_path.endswith('.zip'):
                    model_type = "sb3"
                elif model_path.endswith('.pkl'):
                    model_type = "custom"
                elif model_path.endswith(('.pt', '.pth')):
                    model_type = "pytorch"
                else:
                    print(f"❌ Unknown model format: {model_path}")
                    return None
            
            # Load based on type
            if model_type == "sb3" and SB3_AVAILABLE:
                return self._load_sb3_model(model_path)
            elif model_type == "custom":
                return self._load_custom_model(model_path)
            elif model_type == "pytorch" and TORCH_AVAILABLE:
                return self._load_pytorch_model(model_path)
            else:
                print(f"❌ Model type '{model_type}' not supported or dependencies missing")
                return None
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
    
    def _load_sb3_model(self, model_path):
        """Load Stable-Baselines3 model"""
        try:
            # Try to determine algorithm from filename
            if 'ppo' in model_path.lower():
                model = PPO.load(model_path)
            elif 'a2c' in model_path.lower():
                model = A2C.load(model_path)
            elif 'dqn' in model_path.lower():
                model = DQN.load(model_path)
            else:
                # Default to PPO
                model = PPO.load(model_path)
            
            print(f"✅ Loaded SB3 model: {model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load SB3 model: {e}")
            return None
    
    def _load_custom_model(self, model_path):
        """Load custom model (pickle format)"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            print(f"✅ Loaded custom model: {model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load custom model: {e}")
            return None
    
    def _load_pytorch_model(self, model_path):
        """Load PyTorch model"""
        try:
            if not TORCH_AVAILABLE:
                print("❌ PyTorch not available")
                return None
            
            # This would need to match your model architecture
            # For now, just load the state dict
            model_data = torch.load(model_path, map_location='cpu')
            print(f"✅ Loaded PyTorch model: {model_path}")
            return model_data
            
        except Exception as e:
            print(f"❌ Failed to load PyTorch model: {e}")
            return None
    
    def get_model_info(self, model_path):
        """Get information about a trained model"""
        model = self.load_model(model_path)
        if model is None:
            return None
        
        info = {
            'path': model_path,
            'file_size_mb': round(os.path.getsize(model_path) / (1024 * 1024), 2),
            'loaded': True
        }
        
        # Add framework-specific info
        if SB3_AVAILABLE and isinstance(model, (PPO, A2C, DQN)):
            info.update({
                'framework': 'Stable-Baselines3',
                'algorithm': type(model).__name__,
                'policy': str(model.policy),
                'num_parameters': sum(p.numel() for p in model.policy.parameters())
            })
        elif isinstance(model, dict) and 'q_table' in model:
            info.update({
                'framework': 'Custom Q-Learning',
                'q_table_shape': model['q_table'].shape,
                'training_episodes': model.get('training_episodes', 'Unknown')
            })
        else:
            info['framework'] = 'Unknown/Custom'
        
        return info
    
    def compare_models(self, model_paths):
        """Compare multiple models"""
        comparison = {}
        
        for path in model_paths:
            model_name = os.path.basename(path)
            comparison[model_name] = self.get_model_info(path)
        
        return comparison

def test_model_loading():
    """Test function for model loading"""
    loader = ModelLoader()
    
    print("📁 Available Models:")
    models = loader.list_available_models()
    
    if not models:
        print("  No models found")
        return
    
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['name']} ({model['size_mb']} MB)")
    
    # Test loading first model
    if models:
        first_model = models[0]
        print(f"\n🧪 Testing model loading: {first_model['name']}")
        
        model_info = loader.get_model_info(first_model['path'])
        if model_info:
            print("✅ Model info:")
            for key, value in model_info.items():
                print(f"   {key}: {value}")

if __name__ == "__main__":
    test_model_loading()