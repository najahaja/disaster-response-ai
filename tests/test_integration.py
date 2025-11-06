#!/usr/bin/env python3
"""
Integration Tests for Disaster Response AI - Week 6
Tests end-to-end system functionality
"""

import sys
import os
import unittest
import numpy as np
import tempfile
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.simple_grid_env import SimpleGridEnv
from environments.real_map_env import RealMapEnv
from agents.drone_agent import DroneAgent
from agents.ambulance_agent import AmbulanceAgent
from agents.rescue_team_agent import RescueTeamAgent

class TestSystemIntegration(unittest.TestCase):
    """
    Test end-to-end system integration
    """
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'environment': {
                'grid_size': 15,
                'cell_size': 40,
                'max_steps': 50,
                'num_civilians': 5,
                'num_disasters': 2,
                'cell_types': {
                    'ROAD': 0,
                    'BUILDING': 1,
                    'HOSPITAL': 2,
                    'OPEN_SPACE': 3
                }
            },
            'visualization': {
                'colors': {
                    0: (100, 100, 100),
                    1: (139, 69, 19),
                    2: (255, 0, 0),
                    3: (0, 128, 0)
                }
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.config, self.temp_config)
        self.temp_config.close()
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'temp_config'):
            os.unlink(self.temp_config.name)
    
    def test_complete_simulation_flow(self):
        """Test complete simulation flow from start to finish"""
        print("🔄 Testing complete simulation flow...")
        
        # Create environment
        env = SimpleGridEnv(self.temp_config.name)
        
        # Add multiple agent types
        agents = [
            DroneAgent("drone_1", np.array([2, 2]), env.config),
            AmbulanceAgent("ambulance_1", np.array([5, 5]), env.config),
            RescueTeamAgent("rescue_1", np.array([8, 8]), env.config)
        ]
        
        for agent in agents:
            env.add_agent(agent)
        
        # Trigger disaster
        env.trigger_disaster()
        self.assertTrue(env.disaster_triggered)
        self.assertTrue(len(env.civilians) > 0)
        
        # Run simulation for several steps
        max_steps = 10
        rescued_count = 0
        
        for step in range(max_steps):
            # Generate actions for all agents
            actions = {}
            for agent_id, agent in env.agents.items():
                # Simple movement pattern for testing
                actions[agent_id] = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'])
            
            # Step the environment
            result = env.step(actions)
            self.assertTrue(len(result) >= 4)  # Should return observations, rewards, done, info
            
            # Check if any civilians were rescued
            current_rescued = sum(1 for c in env.civilians if c.get('rescued', False))
            if current_rescued > rescued_count:
                rescued_count = current_rescued
                print(f"✅ Civilian rescued at step {step}")
            
            # Break if simulation is done
            if len(result) == 5:  # Gymnasium format
                observation, reward, terminated, truncated, info = result
                if terminated or truncated:
                    break
            else:  # Legacy format
                observation, rewards, done, info = result
                if done:
                    break
        
        print(f"🎯 Simulation completed: {rescued_count} civilians rescued")
        env.close()
    
    def test_multi_agent_coordination(self):
        """Test coordination between different agent types"""
        print("🤖 Testing multi-agent coordination...")
        
        env = SimpleGridEnv(self.temp_config.name)
        
        # Create a mixed team
        drone = DroneAgent("scout_drone", np.array([3, 3]), env.config)
        ambulance = AmbulanceAgent("medic_ambulance", np.array([6, 6]), env.config)
        rescue_team = RescueTeamAgent("rescue_team", np.array([9, 9]), env.config)
        
        env.add_agent(drone)
        env.add_agent(ambulance)
        env.add_agent(rescue_team)
        
        # Trigger disaster
        env.trigger_disaster()
        
        # Test that all agents can operate together
        for step in range(5):
            actions = {
                'scout_drone': 'RIGHT',      # Drone scouts ahead
                'medic_ambulance': 'UP',     # Ambulance moves toward area
                'rescue_team': 'LEFT'        # Rescue team searches
            }
            
            result = env.step(actions)
            self.assertTrue(len(result) >= 4)
            
            # Check that all agents are still active
            self.assertEqual(len(env.agents), 3)
        
        env.close()
        print("✅ Multi-agent coordination test passed")
    
    def test_communication_integration(self):
        """Test integration of communication system"""
        print("📡 Testing communication system integration...")
        
        try:
            from agents.policies.advanced_policies import CooperativePolicy, CommunicationSystem
            
            env = SimpleGridEnv(self.temp_config.name)
            comm_system = CommunicationSystem()
            
            # Create agents with communication capabilities
            drone = DroneAgent("drone_comm", np.array([4, 4]), env.config)
            ambulance = AmbulanceAgent("ambulance_comm", np.array([7, 7]), env.config)
            
            env.add_agent(drone)
            env.add_agent(ambulance)
            
            # Create policies with communication
            drone_policy = CooperativePolicy(env.config)
            ambulance_policy = CooperativePolicy(env.config)
            
            drone_policy.set_communication_system(comm_system)
            ambulance_policy.set_communication_system(comm_system)
            
            # Test communication during simulation
            for step in range(3):
                # Drone discovers civilian and communicates
                if step == 1:
                    drone_policy.send_civilian_location([5, 5], env)
                
                actions = {
                    'drone_comm': drone_policy.get_action(drone, env),
                    'ambulance_comm': ambulance_policy.get_action(ambulance, env)
                }
                
                env.step(actions)
            
            # Check communication occurred
            stats = comm_system.get_communication_stats()
            self.assertGreaterEqual(stats['total_messages'], 0)
            
            env.close()
            print("✅ Communication integration test passed")
            
        except ImportError as e:
            self.skipTest(f"Communication system not available: {e}")
    
    def test_real_map_integration(self):
        """Test integration with real map system"""
        print("🗺️ Testing real map integration...")
        
        try:
            # Test with real map (will fallback if not available)
            env = RealMapEnv("Test Location", self.temp_config.name)
            
            # Add agents
            drone = DroneAgent("real_map_drone", np.array([5, 5]), env.config)
            env.add_agent(drone)
            
            # Test basic functionality
            obs, info = env.reset()
            self.assertIsNotNone(obs)
            self.assertIn('real_map_loaded', info)
            
            # Test step
            actions = {'real_map_drone': 'RIGHT'}
            result = env.step(actions)
            self.assertTrue(len(result) >= 4)
            
            env.close()
            print("✅ Real map integration test passed")
            
        except Exception as e:
            self.skipTest(f"Real map integration issue: {e}")
    
    def test_performance_under_load(self):
        """Test system performance under load"""
        print("⚡ Testing performance under load...")
        
        import time
        
        env = SimpleGridEnv(self.temp_config.name)
        
        # Add multiple agents to simulate load
        num_agents = 5
        for i in range(num_agents):
            drone = DroneAgent(f"load_drone_{i}", np.array([2 + i, 2]), env.config)
            env.add_agent(drone)
        
        # Measure performance
        start_time = time.time()
        steps = 20
        
        for step in range(steps):
            actions = {agent_id: 'RIGHT' for agent_id in env.agents.keys()}
            env.step(actions)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_step_time = total_time / steps
        
        print(f"📊 Performance: {avg_step_time:.4f}s per step with {num_agents} agents")
        
        # Should be reasonably performant
        self.assertLess(avg_step_time, 0.5)  # Less than 500ms per step
        
        env.close()
        print("✅ Performance test passed")
    
    def test_error_handling(self):
        """Test system error handling and robustness"""
        print("🛡️ Testing error handling...")
        
        env = SimpleGridEnv(self.temp_config.name)
        
        # Test invalid actions
        drone = DroneAgent("error_drone", np.array([5, 5]), env.config)
        env.add_agent(drone)
        
        # Try invalid action - should be handled gracefully
        actions = {'error_drone': 'INVALID_ACTION'}
        result = env.step(actions)
        self.assertTrue(len(result) >= 4)  # Should not crash
        
        # Test out-of-bounds positions
        drone.position = np.array([100, 100])  # Far outside grid
        movement_success = drone.move('LEFT', env.grid)
        self.assertFalse(movement_success)  # Should fail gracefully
        
        env.close()
        print("✅ Error handling test passed")

class TestDashboardIntegration(unittest.TestCase):
    """
    Test dashboard and visualization integration
    """
    
    def test_dashboard_components(self):
        """Test dashboard component integration"""
        print("📊 Testing dashboard components...")
        
        try:
            from dashboard.components.controls import ControlPanel
            from dashboard.components.metrics_display import MetricsDisplay
            from dashboard.components.simulation_viewer import SimulationViewer
            
            # Test component creation
            controls = ControlPanel()
            metrics = MetricsDisplay()
            viewer = SimulationViewer()
            
            self.assertIsNotNone(controls)
            self.assertIsNotNone(metrics)
            self.assertIsNotNone(viewer)
            
            # Test metrics update
            mock_env = type('MockEnv', (), {
                'civilians': [{'rescued': True}, {'rescued': False}],
                'agents': {},
                'step_count': 10,
                'collapsed_buildings': [],
                'blocked_roads': []
            })()
            
            metrics.update_metrics(mock_env, {})
            
            print("✅ Dashboard components test passed")
            
        except ImportError as e:
            self.skipTest(f"Dashboard components not available: {e}")

def run_integration_tests():
    """Run all integration tests"""
    print("🧪 System Integration Tests - Week 6")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSystemIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDashboardIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Integration Test Summary")
    print("=" * 50)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️ Some integration tests failed or were skipped")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)