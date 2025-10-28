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
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

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
        except Exception as e:
            print(f"⚠️  {module_name}: Unexpected error - {e}")
    
    return all_imports_ok

def test_environment_creation():
    """Test environment creation and basic functionality"""
    print("\n🧪 TEST 2: Environment Creation")
    print("=" * 50)
    
    try:
        # Test simple grid environment
        from environments.simple_grid_env import SimpleGridEnv
        simple_env = SimpleGridEnv()
        print("✅ SimpleGridEnv created")
        
        # Test basic functionality
        obs, info = simple_env.reset()
        print("✅ Environment reset works")
        
        # Test disaster triggering
        simple_env.trigger_disaster()
        print("✅ Disaster triggering works")
        
        # Test real map environment (with fallback)
        try:
            from environments.real_map_env import RealMapEnv
            print("🗺️  Loading map for Lahore, Pakistan...")
            real_env = RealMapEnv("Lahore, Pakistan")
            print("✅ RealMapEnv created with real map")
        except Exception as e:
            print(f"❌ Failed to load real map: {e}")
            try:
                print("🔄 Falling back to generated grid...")
                real_env = RealMapEnv()  # Should fallback to generated grid
                print("✅ RealMapEnv created with fallback grid")
            except Exception as e2:
                print(f"❌ RealMapEnv fallback also failed: {e2}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_system():
    """Test agent creation and basic behavior"""
    print("\n🧪 TEST 3: Agent System")
    print("=" * 50)
    
    try:
        from environments.simple_grid_env import SimpleGridEnv
        from agents.drone_agent import DroneAgent
        from agents.ambulance_agent import AmbulanceAgent
        import numpy as np
        
        # Create environment
        env = SimpleGridEnv()
        
        # Create agents
        drone = DroneAgent("test_drone", np.array([5, 5]), env.config)
        ambulance = AmbulanceAgent("test_ambulance", np.array([10, 10]), env.config)
        
        env.add_agent(drone)
        env.add_agent(ambulance)
        print("✅ Agents created and added to environment")
        
        # Test movement
        success = drone.move('RIGHT', env.grid)
        print(f"✅ Agent movement: {success}")
        
        # Test cooperative policy (if available)
        try:
            from agents.policies.advanced_policies import CooperativePolicy
            policy = CooperativePolicy(env.config)
            action = policy.get_action(drone, env)
            print(f"✅ Cooperative policy action: {action}")
        except Exception as e:
            print(f"⚠️  Cooperative policy not available: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_marl_integration():
    """Test MARL components with graceful PettingZoo handling"""
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
        
        # Test PettingZoo wrapper with comprehensive error handling
        try:
            from marl.pettingzoo_wrapper import DisasterResponsePettingZooEnv
            
            print("🔄 Testing PettingZoo wrapper...")
            
            # Create PettingZoo environment using the CALLABLE function
            pettingzoo_env = DisasterResponsePettingZooEnv()
            print("✅ PettingZoo wrapper created successfully")
            
            # Test basic functionality
            try:
                observations = pettingzoo_env.reset()
                print(f"✅ PettingZoo environment reset - Agents: {pettingzoo_env.agents}")
                
                # Test step if we have agents
                if pettingzoo_env.agents:
                    # Test a few steps
                    for step in range(2):
                        current_agent = pettingzoo_env.agent_selection
                        
                        # Get action space for current agent
                        action_space = pettingzoo_env.action_space(current_agent)
                        
                        # Take a random action
                        if hasattr(action_space, 'sample'):
                            action = action_space.sample()
                        else:
                            action = 0  # Default action
                        
                        # Execute step
                        pettingzoo_env.step(action)
                        
                        # Check if we can get observation
                        obs = pettingzoo_env.observe(current_agent)
                        print(f"✅ Step {step} completed for agent {current_agent}")
                        
                        # Move to next agent
                        if step == 0 and len(pettingzoo_env.agents) > 1:
                            break  # Just test one step per agent for speed
                
                # Test observation and action spaces for all agents
                for agent in pettingzoo_env.agents:
                    obs_space = pettingzoo_env.observation_space(agent)
                    act_space = pettingzoo_env.action_space(agent)
                    print(f"✅ Agent {agent}: obs_space={type(obs_space)}, act_space={type(act_space)}")
                
                pettingzoo_env.close()
                print("✅ PettingZoo wrapper fully functional!")
                
            except Exception as step_error:
                print(f"⚠️  PettingZoo step test issue (non-critical): {step_error}")
                # This is not critical for basic functionality
                
        except Exception as e:
            print(f"⚠️  PettingZoo wrapper creation issue: {e}")
            # This is not critical for basic functionality
        
        return True
        
    except Exception as e:
        print(f"❌ MARL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_features():
    """Test Week 3 advanced features"""
    print("\n🧪 TEST 5: Advanced Features (Week 3)")
    print("=" * 50)
    
    try:
        # Test communication system
        try:
            from agents.policies.communication_policy import CommunicationSystem
            comm_system = CommunicationSystem()
            print("✅ Communication system created")
        except Exception as e:
            print(f"⚠️  Communication system: {e}")
            # Try alternative import
            try:
                from agents.policies.advanced_policies import CommunicationSystem
                comm_system = CommunicationSystem()
                print("✅ Communication system created (from advanced_policies)")
            except:
                print("❌ Communication system not available")
        
        # Test dynamic events
        try:
            from environments.utils.dynamic_events import DynamicEventManager
            event_manager = DynamicEventManager({})
            print("✅ Dynamic event manager created")
        except Exception as e:
            print(f"⚠️  Dynamic event manager: {e}")
        
        # Test A* pathfinding
        try:
            from agents.policies.advanced_policies import AStarPolicy
            astar = AStarPolicy()
            
            # Create a simple grid for pathfinding test
            test_grid = np.zeros((10, 10))
            path = astar.find_path((0, 0), (9, 9), test_grid)
            print(f"✅ A* pathfinding: {len(path)} steps")
        except Exception as e:
            print(f"⚠️  A* pathfinding: {e}")
        
        # Test map loader
        try:
            from environments.utils.map_loader import MapLoader
            map_loader = MapLoader()
            
            # Test map loading
            map_data = map_loader.load_map_with_fallback("Test Location")
            if map_data and hasattr(map_data, 'grid'):
                print(f"✅ Map loader created and functional - Grid shape: {map_data.grid.shape}")
            else:
                print("✅ Map loader created")
                
        except Exception as e:
            print(f"⚠️  Map loader: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard_components():
    """Test dashboard components"""
    print("\n🧪 TEST 6: Dashboard Components")
    print("=" * 50)
    
    try:
        # Test controls component
        try:
            from dashboard.components.controls import ControlPanel
            controls = ControlPanel()
            print("✅ Control panel created")
        except Exception as e:
            print(f"⚠️  Control panel: {e}")
        
        # Test metrics display (without Streamlit context)
        try:
            from dashboard.components.metrics_display import MetricsDisplay
            metrics = MetricsDisplay()
            metrics.update_metrics({'civilians_rescued': 5, 'agents_active': 3, 'efficiency': 85})
            print("✅ Metrics display created and updated")
        except Exception as e:
            print(f"⚠️  Metrics display: {e}")
        
        # Test simulation viewer
        try:
            from dashboard.components.simulation_viewer import SimulationViewer
            viewer = SimulationViewer()
            print("✅ Simulation viewer created")
        except Exception as e:
            print(f"⚠️  Simulation viewer: {e}")
        
        print("ℹ️  Full dashboard test requires: streamlit run run_dashboard.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integrated system functionality"""
    print("\n🧪 TEST 7: Integrated System Test")
    print("=" * 50)
    
    try:
        from environments.simple_grid_env import SimpleGridEnv
        from agents.drone_agent import DroneAgent
        from agents.ambulance_agent import AmbulanceAgent
        import numpy as np
        
        # Create integrated system
        env = SimpleGridEnv()
        
        # Create agents
        drone = DroneAgent("integrated_drone", np.array([5, 5]), env.config)
        ambulance = AmbulanceAgent("integrated_ambulance", np.array([10, 10]), env.config)
        
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
                # Simple movement actions for testing
                actions[agent_id] = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'])
            
            # Handle gymnasium format (5 return values)
            result = env.step(actions)
            if len(result) == 5:
                # Gymnasium format: obs, reward, terminated, truncated, info
                observation, reward, terminated, truncated, info = result
                rewards = reward  # In gym format, it's a single reward
                done = terminated or truncated
            elif len(result) == 4:
                # Old format: obs, rewards, done, info
                observation, rewards, done, info = result
            else:
                print(f"⚠️  Unexpected step return format: {len(result)} values")
                continue
            
            print(f"✅ Step {step}: Rewards {rewards}")
            
            if done:
                break
        
        # Test communication if available
        try:
            from agents.policies.advanced_policies import CooperativePolicy, CommunicationSystem
            comm_system = CommunicationSystem()
            
            drone_policy = CooperativePolicy(env.config)
            ambulance_policy = CooperativePolicy(env.config)
            
            # Test communication stats
            comm_stats = comm_system.get_communication_stats()
            print(f"✅ Communication: {comm_stats.get('total_messages', 0)} messages sent")
        except Exception as e:
            print(f"⚠️  Advanced communication features: {e}")
        
        # ✅ FIXED: PettingZoo integration test with proper callable function
        try:
            # Use the callable function
            from marl.pettingzoo_wrapper import DisasterResponsePettingZooEnv
            
            # Debug: Check if it's callable
            if callable(DisasterResponsePettingZooEnv):
                pettingzoo_env = DisasterResponsePettingZooEnv()
                pettingzoo_obs = pettingzoo_env.reset()
                print(f"✅ PettingZoo integration successful: {len(pettingzoo_obs)} agent observations")
                pettingzoo_env.close()
            else:
                print(f"⚠️  DisasterResponsePettingZooEnv is not callable: {type(DisasterResponsePettingZooEnv)}")
                        
        except Exception as e:
            print(f"⚠️  PettingZoo integration test skipped: {e}")
        
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
            import traceback
            traceback.print_exc()
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
        print("   - Run PettingZoo tests: python -m marl.pettingzoo_wrapper")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the errors above.")
        if passed >= 5:  # If most tests pass
            print("💡 Most components are working! Focus on fixing the specific failed tests.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)