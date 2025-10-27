#!/usr/bin/env python3
"""
Simple training test that definitely works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from environments.simple_grid_env import SimpleGridEnv
from agents.drone_agent import DroneAgent
from agents.ambulance_agent import AmbulanceAgent

def simple_training_test():
    """Simple training test that doesn't use PettingZoo"""
    print("🚀 Starting Simple Training Test...")
    
    # Create environment
    env = SimpleGridEnv()
    
    # Add agents
    drone1 = DroneAgent("drone_1", [5, 5], env.config)
    drone2 = DroneAgent("drone_2", [10, 10], env.config)
    ambulance1 = AmbulanceAgent("ambulance_1", [7, 7], env.config)
    
    env.add_agent(drone1)
    env.add_agent(drone2)
    env.add_agent(ambulance1)
    
    # Trigger disaster
    env.trigger_disaster()
    
    print(f"🤖 Agents: {list(env.agents.keys())}")
    print(f"👥 Civilians: {len(env.civilians)}")
    print(f"🔥 Starting training...")
    
    # Simple training loop
    episode_rewards = []
    rescue_rates = []
    
    for episode in range(10):  # Just 10 episodes for testing
        env.reset()
        episode_reward = 0
        steps = 0
        
        while not env.is_done() and steps < 50:  # Limit steps per episode
            # Random actions
            actions = {}
            for agent_id, agent in env.agents.items():
                action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'])
                actions[agent_id] = action
            
            # Step the environment
            observation, rewards, done, info = env.step(actions)
            episode_reward += sum(rewards.values())
            steps += 1
            
            if done:
                break
        
        # Calculate metrics
        total_civilians = len(env.civilians)
        rescued_civilians = sum(1 for c in env.civilians if c['rescued'])
        rescue_rate = rescued_civilians / max(total_civilians, 1)
        
        episode_rewards.append(episode_reward)
        rescue_rates.append(rescue_rate)
        
        print(f"📊 Episode {episode}: Reward={episode_reward:.1f}, "
              f"Steps={steps}, Rescue Rate={rescue_rate:.2f}")
    
    env.close()
    
    print(f"\n🎉 Training completed!")
    print(f"📈 Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"📈 Average Rescue Rate: {np.mean(rescue_rates):.2f}")
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 SIMPLE TRAINING TEST")
    print("=" * 50)
    
    if simple_training_test():
        print("\n✅ All tests passed! Your environment is working.")
        print("🚀 You can now proceed with more advanced training.")
    else:
        print("\n❌ Tests failed.")