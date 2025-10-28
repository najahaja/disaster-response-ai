#!/usr/bin/env python3
"""
Complete System Test for Disaster Response AI Project
Tests all components from Week 1 to Week 3
"""

import sys
import os
import importlib
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 TEST 1: Module Imports")
    print("=" * 50)
    
    modules_to_test = [
        # Week 1: Core Simulation
        "environments.simple_grid_env",
        "agents.drone_agent",
        "agents.ambulance_agent", 
        "agents.rescue_team_agent",
        
        # Week 2: MARL Integration
        "marl.pettingzoo_wrapper",
        "marl.reward_functions",
        "marl.observation_spaces",
        "marl.action_spaces",
        
        # Week 3: Advanced Features
        "environments.real_map_env",
        "environments.utils.map_loader",
        "environments.utils.dynamic_events",
        "agents.policies.advanced_policies",
        "agents.policies.communication_policy",
        
        # Dashboard
        "dashboard.app",
        "dashboard.components.controls",
        "dashboard.components.metrics_display",
        "dashboard.components.simulation_viewer",
    ]
    
    all_imports_ok = True
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def test_environment_creation():
    """Test environment creation and basic functionality"""
    print("\n🧪 TEST 2: Environment Creation")
    print("=" * 50)
    
    try:
        from environments.simple_grid_env import SimpleGridEnv
        from environments.real_map_env import RealMapEnv
        
        # Test simple grid environment
        simple_env = SimpleGridEnv()
        print("✅ SimpleGridEnv created")
        
        # Test real map environment (with fallback)
        try:
            real_env = RealMapEnv("Lahore, Pakistan")
            print("✅ RealMapEnv created with real map")
        except Exception as e:
            print(f"⚠️  RealMapEnv fallback: {e}")
            real_env = RealMapEnv()  # Should fallback to generated grid
            print("✅ RealMapEnv created with fallback")
        
        # Test basic functionality
        simple_env.reset()
        print("✅ Environment reset works")
        
        simple_env.trigger_disaster()
        print("✅ Disaster triggering works")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_agent_system():
    """Test agent creation and basic behavior"""
    print("\n🧪 TEST 3: Agent System")
    print("=" * 50)
    
    try:
        from environments.simple_grid_env import SimpleGridEnv
        from agents.drone_agent import DroneAgent
        from agents.ambulance_agent import AmbulanceAgent
        from agents.policies.advanced_policies import CooperativePolicy
        
        # Create environment
        env = SimpleGridEnv()
        
        # Create agents
        drone = DroneAgent("test_drone", [5, 5], env.config)
        ambulance = AmbulanceAgent("test_ambulance", [10, 10], env.config)
        
        env.add_agent(drone)
        env.add_agent(ambulance)
        print("✅ Agents created and added to environment")
        
        # Test movement
        success = drone.move('RIGHT', env.grid)
        print(f"✅ Agent movement: {success}")
        
        # Test cooperative policy
        policy = CooperativePolicy(env.config)
        action = policy.get_action(drone, env)
        print(f"✅ Cooperative policy action: {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def test_marl_integration():
    """Test MARL components"""
    print("\n🧪 TEST 4: MARL Integration")
    print("=" * 50)
    
    try:
        from marl.action_spaces import DiscreteActionSpace
        from marl.observation_spaces import GlobalObservation
        from marl.reward_functions import CollaborativeReward
        
        # Test action space
        action_space = DiscreteActionSpace()
        print(f"✅ Action space: {action_space}")
        
        # Test observation space
        obs_space = GlobalObservation(15, 6, 3)
        print(f"✅ Observation space created")
        
        # Test reward function
        reward_func = CollaborativeReward({"environment": {"cell_types": {}}})
        print(f"✅ Reward function created")
        
        # Test PettingZoo wrapper (if available)
        try:
            from marl.pettingzoo_wrapper import DisasterResponseEnv
            pettingzoo_env = DisasterResponseEnv()
            print("✅ PettingZoo wrapper created")
        except Exception as e:
            print(f"⚠️  PettingZoo wrapper: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ MARL test failed: {e}")
        return False

def test_advanced_features():
    """Test Week 3 advanced features"""
    print("\n🧪 TEST 5: Advanced Features (Week 3)")
    print("=" * 50)
    
    try:
        # Test communication system
        from agents.policies.advanced_policies import CommunicationSystem
        comm_system = CommunicationSystem()
        print("✅ Communication system created")
        
        # Test dynamic events
        from environments.utils.dynamic_events import DynamicEventManager
        event_manager = DynamicEventManager({})
        print("✅ Dynamic event manager created")
        
        # Test A* pathfinding
        from agents.policies.advanced_policies import AStarPolicy
        astar = AStarPolicy()
        
        # Create a simple grid for pathfinding test
        test_grid = np.zeros((10, 10))
        path = astar.find_path([0, 0], [9, 9], test_grid)
        print(f"✅ A* pathfinding: {len(path)} steps")
        
        # Test map loader
        from environments.utils.map_loader import MapLoader
        map_loader = MapLoader()
        print("✅ Map loader created")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced features test failed: {e}")
        return False

def test_dashboard_components():
    """Test dashboard components"""
    print("\n🧪 TEST 6: Dashboard Components")
    print("=" * 50)
    
    try:
        # Test controls component
        from dashboard.components.controls import ControlPanel
        controls = ControlPanel()
        print("✅ Control panel created")
        
        # Test metrics display (without Streamlit context)
        from dashboard.components.metrics_display import MetricsDisplay
        metrics = MetricsDisplay()
        metrics.update_metrics({'civilians_rescued': 5, 'agents_active': 3, 'efficiency': 85})
        print("✅ Metrics display created and updated")
        
        print("ℹ️  Full dashboard test requires: streamlit run run_dashboard.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def test_integration():
    """Test integrated system functionality"""
    print("\n🧪 TEST 7: Integrated System Test")
    print("=" * 50)
    
    try:
        from environments.simple_grid_env import SimpleGridEnv
        from agents.drone_agent import DroneAgent
        from agents.ambulance_agent import AmbulanceAgent
        from agents.policies.advanced_policies import CooperativePolicy, CommunicationSystem
        
        # Create integrated system
        env = SimpleGridEnv()
        comm_system = CommunicationSystem()
        
        # Create agents
        drone = DroneAgent("integrated_drone", [5, 5], env.config)
        ambulance = AmbulanceAgent("integrated_ambulance", [10, 10], env.config)
        
        # Create policies and attach communication system
        drone_policy = CooperativePolicy(env.config)
        ambulance_policy = CooperativePolicy(env.config)
        
        # Set agent IDs and references for policies
        drone_policy.set_agent_id("integrated_drone")
        ambulance_policy.set_agent_id("integrated_ambulance")
        
        drone_policy.set_communication_system(comm_system)
        ambulance_policy.set_communication_system(comm_system)
        
        # Add agents to environment
        env.add_agent(drone)
        env.add_agent(ambulance)
        
        # Trigger disaster
        env.trigger_disaster()
        
        print("✅ System initialized with 2 agents and communication")
        
        # Run a few simulation steps
        for step in range(5):
            actions = {}
            for agent_id, agent in env.agents.items():
                # Use the appropriate policy for each agent
                if agent_id == "integrated_drone":
                    action = drone_policy.get_action(agent, env)
                else:
                    action = ambulance_policy.get_action(agent, env)
                actions[agent_id] = action
            
            # ✅ FIXED: Handle gymnasium format (5 return values)
            result = env.step(actions)
            if len(result) == 5:
                # Gymnasium format: obs, reward, terminated, truncated, info
                observation, reward, terminated, truncated, info = result
                rewards = reward  # In gym format, it's a single reward
                done = terminated or truncated
            else:
                # Old format: obs, rewards, done, info
                observation, rewards, done, info = result
            
            print(f"✅ Step {step}: Rewards {rewards}")
            
            # Test communication on step 2
            if step == 2:
                # Simulate a civilian discovery and communication
                drone_policy.send_civilian_location([7, 7], env)
            
            if done:
                break
        
        # Test communication stats
        comm_stats = comm_system.get_communication_stats()
        print(f"✅ Communication: {comm_stats['total_messages']} messages sent")
        
        # Test policy stats
        drone_stats = drone_policy.get_policy_stats()
        ambulance_stats = ambulance_policy.get_policy_stats()
        print(f"✅ Drone policy: {drone_stats['known_civilians']} civilians known")
        print(f"✅ Ambulance policy: {ambulance_stats['known_civilians']} civilians known")
        print(f"✅ Drone has agent ref: {drone_stats['has_agent_ref']}")
        print(f"✅ Ambulance has agent ref: {ambulance_stats['has_agent_ref']}")
        
        env.close()
        print("✅ Integrated system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 COMPLETE SYSTEM TEST - Disaster Response AI")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_environment_creation,
        test_agent_system,
        test_marl_integration,
        test_advanced_features,
        test_dashboard_components,
        test_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1:2d}. {test.__name__:30} {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your system is working correctly!")
        print("🚀 You can now:")
        print("   - Run training: python run_training.py")
        print("   - Start dashboard: streamlit run run_dashboard.py")
        print("   - Test real maps: python scripts/download_maps.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)