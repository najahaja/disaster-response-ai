#!/usr/bin/env python3
"""
Simple test for Week 2 components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test if we can import all Week 2 modules"""
    print("🧪 Testing Week 2 Imports...")
    
    try:
        from marl.action_spaces import DiscreteActionSpace
        print("✅ action_spaces imported")
        
        from marl.observation_spaces import GlobalObservation
        print("✅ observation_spaces imported")
        
        from marl.reward_functions import CollaborativeReward
        print("✅ reward_functions imported")
        
        # Test config loading
        import yaml
        with open("training/configs/base_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print("✅ config files loaded")
        
        print("🎉 All Week 2 imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_simple_training():
    """Test a simple training loop without PettingZoo complications"""
    print("\n🧪 Testing Simple Training...")
    
    try:
        from environments.simple_grid_env import SimpleGridEnv
        from agents.drone_agent import DroneAgent
        from agents.ambulance_agent import AmbulanceAgent
        
        # Create simple environment
        env = SimpleGridEnv()
        
        # Add agents
        drone = DroneAgent("test_drone", [5, 5], env.config)
        ambulance = AmbulanceAgent("test_ambulance", [10, 10], env.config)
        env.add_agent(drone)
        env.add_agent(ambulance)
        
        # Trigger disaster
        env.trigger_disaster()
        
        # Simple training loop
        for step in range(10):
            actions = {
                "test_drone": "RIGHT",
                "test_ambulance": "LEFT"
            }
            obs, rewards, done, info = env.step(actions)
            print(f"Step {step}: Rewards {rewards}")
            
            if done:
                break
        
        env.close()
        print("✅ Simple training test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Simple training failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Week 2 Simple Test")
    print("=" * 50)
    
    if test_basic_imports() and test_simple_training():
        print("\n🎉 Week 2 foundation is working!")
        print("🚀 You can now proceed with MARL integration")
    else:
        print("\n❌ Some tests failed. Check the imports and dependencies.")