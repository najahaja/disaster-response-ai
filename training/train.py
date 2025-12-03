import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.simple_grid_env import SimpleGridEnv
from agents.drone_agent import DroneAgent
from agents.ambulance_agent import AmbulanceAgent
from agents.rescue_team_agent import RescueTeamAgent

# --- 1. Special Wrapper for Training ---
# This fixes the "Empty Map" problem by adding agents automatically on reset.
class TrainingEnv(SimpleGridEnv):
    def reset(self, *, seed=None, options=None):
        # 1. Reset the grid (clears everything)
        obs, info = super().reset(seed=seed, options=options)
        
        # 2. Add Agents (We need to do this manually for training)
        # Drones
        for i in range(self.config['agents']['drone']['count']):
            self.add_agent(DroneAgent(f"drone_{i}", [2+i, 2], self.config))
            
        # Ambulances
        for i in range(self.config['agents']['ambulance']['count']):
            self.add_agent(AmbulanceAgent(f"ambulance_{i}", [5, 5+i], self.config))
            
        # Rescue Teams
        for i in range(self.config['agents']['rescue_team']['count']):
            self.add_agent(RescueTeamAgent(f"rescue_{i}", [8, 8+i], self.config))
            
        # 3. Trigger Disaster (To create civilians to rescue!)
        self.trigger_disaster()
        
        # 4. Get new observation with agents and civilians visible
        return self._get_gym_observation(), info

# --- 2. Custom CNN for 45x45 Grid ---
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            # Layer 1: 45x45 -> 10x10
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Layer 2: 10x10 -> 4x4
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Layer 3: 4x4 -> 2x2
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape dynamically
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

def main():
    print("🚀 Starting AI Training for Disaster Response...")

    # --- 3. Create and Wrap the Environment ---
    def make_env():
        # Use our special TrainingEnv that adds agents automatically
        env = TrainingEnv(config_path="config.yaml")
        return Monitor(env)

    env = DummyVecEnv([make_env])

    # --- 4. Setup the PPO Model ---
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        policy_kwargs={"features_extractor_class": CustomCNN}
    )

    # --- 5. Train! ---
    TRAIN_STEPS = 100000 
    print(f"🧠 Training for {TRAIN_STEPS} steps... (Press Ctrl+C to stop early)")
    
    try:
        model.learn(total_timesteps=TRAIN_STEPS)
        print("✅ Training Finished!")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted manually. Saving current progress...")

    # --- 6. Save the Model ---
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_path = f"{models_dir}/disaster_response_model"
    model.save(model_path)
    print(f"💾 Model saved to {model_path}.zip")

if __name__ == "__main__":
    main()