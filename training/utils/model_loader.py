import os
import pickle
import yaml
from typing import Dict, Any

class ModelLoader:
    """
    Utility class for saving and loading trained models
    """
    
    def __init__(self, base_path="./trained_models"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_model(self, model, model_name, metadata=None):
        """
        Save a trained model
        """
        model_path = os.path.join(self.base_path, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save model weights
        model.save(os.path.join(model_path, "model"))
        
        # Save metadata
        if metadata:
            with open(os.path.join(model_path, "metadata.yaml"), 'w') as f:
                yaml.dump(metadata, f)
        
        print(f"✅ Model saved to {model_path}")
    
    def load_model(self, model_name, model_class=None):
        """
        Load a trained model
        """
        model_path = os.path.join(self.base_path, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        if model_class:
            model = model_class.load(os.path.join(model_path, "model"))
        else:
            # Try to load with common RL frameworks
            try:
                from stable_baselines3 import PPO
                model = PPO.load(os.path.join(model_path, "model"))
            except ImportError:
                raise ImportError("Could not automatically load model. Please specify model_class.")
        
        # Load metadata
        metadata_path = os.path.join(model_path, "metadata.yaml")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
        
        print(f"✅ Model loaded from {model_path}")
        return model, metadata
    
    def list_models(self):
        """List all saved models"""
        return [d for d in os.listdir(self.base_path) 
                if os.path.isdir(os.path.join(self.base_path, d))]