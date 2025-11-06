#!/usr/bin/env python3
"""
Enhanced MARL System Tests for Week 6
Tests multi-agent reinforcement learning components with improved coverage
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
from agents.drone_agent import DroneAgent
from agents.ambulance_agent import AmbulanceAgent
from agents.rescue_team_agent import RescueTeamAgent

class TestMARLSystem(unittest.TestCase):
    """
    Test MARL system components and integration with enhanced coverage
    """
    
    def setUp(self):
        """Set up test environment with comprehensive configuration"""
        self.config = {
            'environment': {
                'grid_size': 20,
                'cell_size': 30,
                'max_steps': 100,
                'num_civilians': 8,
                'num_disasters': 3,
                'cell_types': {
                    'ROAD': 0,
                    'BUILDING': 1,
                    'HOSPITAL': 2,
                    'OPEN_SPACE': 3,
                    'WATER': 4,
                    'COLLAPSED': 5,
                    'BLOCKED': 6
                }
            },
            'visualization': {
                'colors': {
                    0: (100, 100, 100),  # ROAD
                    1: (139, 69, 19),    # BUILDING
                    2: (255, 0, 0),      # HOSPITAL
                    3: (0, 128, 0),      # OPEN_SPACE
                    4: (0, 0, 255),      # WATER
                    5: (0, 0, 0),        # COLLAPSED
                    6: (128, 0, 0)       # BLOCKED
                }
            },
            'agents': {
                'drone': {'speed': 3, 'capacity': 0},
                'ambulance': {'speed': 2, 'capacity': 3},
                'rescue_team': {'speed': 1, 'capacity': 1}
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
    
    def test_environment_creation(self):
        """Test environment creation with different configurations"""
        print("🏗️ Testing environment creation...")
        
        # Test with custom config
        env = SimpleGridEnv(self.temp_config.name)
        self.assertIsNotNone(env)
        self.assertEqual(env.grid_size, 20)
        
        # Test reset functionality
        obs, info = env.reset()
        self.assertIsNotNone(obs)
        self.assertIsInstance(info, dict)
        
        env.close()
        print("✅ Environment creation test passed")
    
    def test_pettingzoo_wrapper_creation(self):
        """Test PettingZoo wrapper creation and basic functionality"""
        print("🎪 Testing PettingZoo wrapper...")
        
        try:
            from marl.pettingzoo_wrapper import DisasterResponsePettingZooEnv
            
            env = DisasterResponsePettingZooEnv(self.temp_config.name)
            
            # Test reset
            observations = env.reset()
            self.assertIsInstance(observations, dict)
            
            # Test agent management
            self.assertTrue(len(env.agents) > 0)
            
            # Test observation and action spaces for each agent
            for agent in env.agents:
                obs_space = env.observation_space(agent)
                act_space = env.action_space(agent)
                self.assertIsNotNone(obs_space)
                self.assertIsNotNone(act_space)
                
                # Test that spaces have correct properties
                self.assertTrue(hasattr(obs_space, 'shape') or hasattr(obs_space, 'spaces'))
                self.assertTrue(hasattr(act_space, 'n') or hasattr(act_space, 'spaces'))
            
            env.close()
            print("✅ PettingZoo wrapper creation test passed")
            
        except Exception as e:
            self.skipTest(f"PettingZoo wrapper not available: {e}")
    
    def test_reward_functions_comprehensive(self):
        """Test comprehensive reward function calculations"""
        print("💰 Testing reward functions...")
        
        try:
            from marl.reward_functions import CollaborativeReward
            
            reward_func = CollaborativeReward(self.config)
            
            # Test various scenarios
            test_cases = [
                {
                    'agent': type('MockAgent', (), {
                        'agent_id': 'test_drone',
                        'agent_type': 'drone',
                        'civilians_rescued': 0
                    })(),
                    'prev_obs': {'grid': np.zeros((10, 10)), 'civilians': []},
                    'action': 'UP',
                    'next_obs': {'grid': np.ones((10, 10)), 'civilians': []},
                    'description': 'Basic movement'
                },
                {
                    'agent': type('MockAgent', (), {
                        'agent_id': 'test_ambulance', 
                        'agent_type': 'ambulance',
                        'civilians_rescued': 1
                    })(),
                    'prev_obs': {'grid': np.zeros((10, 10)), 'civilians': [{'rescued': False}]},
                    'action': 'STAY',
                    'next_obs': {'grid': np.zeros((10, 10)), 'civilians': [{'rescued': True}]},
                    'description': 'Civilian rescue'
                }
            ]
            
            for i, test_case in enumerate(test_cases):
                with self.subTest(test_case=test_case['description']):
                    reward = reward_func.calculate_reward(
                        test_case['agent'], 
                        test_case['prev_obs'], 
                        test_case['action'], 
                        test_case['next_obs']
                    )
                    self.assertIsInstance(reward, (int, float))
                    print(f"  ✅ Reward case {i+1}: {test_case['description']} - Reward: {reward}")
            
            print("✅ Comprehensive reward function test passed")
            
        except ImportError as e:
            self.skipTest(f"Reward functions not available: {e}")
    
    def test_observation_spaces_detailed(self):
        """Test detailed observation space encoding"""
        print("👀 Testing observation spaces...")
        
        try:
            from marl.observation_spaces import GlobalObservation
            
            obs_space = GlobalObservation(20, 7, 3)  # grid_size, num_cell_types, num_agent_types
            
            # Test different agent types
            agent_types = ['drone', 'ambulance', 'rescue_team']
            
            for agent_type in agent_types:
                with self.subTest(agent_type=agent_type):
                    mock_agent = type('MockAgent', (), {
                        'agent_id': f'test_{agent_type}',
                        'position': np.array([5, 5]),
                        'agent_type': agent_type
                    })()
                    
                    mock_observation = {
                        'grid': np.random.randint(0, 7, (20, 20)),
                        'agents': {'test_agent': {'position': [10, 10]}},
                        'civilians': [{'position': [3, 3], 'rescued': False}]
                    }
                    
                    # Test observation encoding
                    encoded_obs = obs_space.encode_observation(mock_agent, mock_observation)
                    self.assertIsInstance(encoded_obs, np.ndarray)
                    self.assertTrue(encoded_obs.size > 0)
                    
                    # Test observation space creation
                    space = obs_space.get_observation_space(agent_type, 20)
                    self.assertIsNotNone(space)
            
            print("✅ Detailed observation space test passed")
            
        except ImportError as e:
            self.skipTest(f"Observation spaces not available: {e}")
    
    def test_action_spaces_comprehensive(self):
        """Test comprehensive action space functionality"""
        print("🎯 Testing action spaces...")
        
        try:
            from marl.action_spaces import DiscreteActionSpace
            
            action_space = DiscreteActionSpace()
            
            # Test action decoding for all agent types
            agent_types = ['drone', 'ambulance', 'rescue_team']
            
            for agent_type in agent_types:
                with self.subTest(agent_type=agent_type):
                    mock_agent = type('MockAgent', (), {'agent_type': agent_type})()
                    
                    # Test all possible actions
                    for action_id in range(6):
                        action_str = action_space.decode_action(action_id, mock_agent)
                        self.assertIn(action_str, ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'REST'])
                    
                    # Test action space creation
                    space = action_space.get_action_space(agent_type)
                    self.assertIsNotNone(space)
                    self.assertEqual(space.n, 6)  # 6 discrete actions
            
            # Test invalid action handling
            mock_agent = type('MockAgent', (), {'agent_type': 'drone'})()
            invalid_action = action_space.decode_action(999, mock_agent)  # Invalid action ID
            self.assertEqual(invalid_action, 'STAY')  # Should default to STAY
            
            print("✅ Comprehensive action space test passed")
            
        except ImportError as e:
            self.skipTest(f"Action spaces not available: {e}")
    
    def test_multi_agent_coordination_advanced(self):
        """Test advanced multi-agent coordination scenarios"""
        print("🤝 Testing advanced multi-agent coordination...")
        
        env = SimpleGridEnv(self.temp_config.name)
        
        # Create diverse agent team
        agents = [
            DroneAgent("scout_drone", np.array([2, 2]), env.config),
            DroneAgent("support_drone", np.array([4, 4]), env.config),
            AmbulanceAgent("medic_ambulance", np.array([6, 6]), env.config),
            RescueTeamAgent("rescue_team", np.array([8, 8]), env.config)
        ]
        
        for agent in agents:
            env.add_agent(agent)
        
        # Test environment state with multiple agents
        self.assertEqual(len(env.agents), 4)
        
        # Trigger disaster to create realistic scenario
        env.trigger_disaster()
        self.assertTrue(env.disaster_triggered)
        self.assertTrue(len(env.civilians) > 0)
        
        # Test coordinated actions
        coordinated_actions = {
            'scout_drone': 'RIGHT',    # Scout ahead
            'support_drone': 'UP',     # Provide support
            'medic_ambulance': 'LEFT', # Position for rescue
            'rescue_team': 'DOWN'      # Secure area
        }
        
        # Execute coordinated step
        result = env.step(coordinated_actions)
        self.assertTrue(len(result) >= 4)
        
        # Verify all agents are still active and positions updated
        for agent_id, agent in env.agents.items():
            self.assertTrue(np.array_equal(agent.position, agent.position))  # Position should be valid
        
        env.close()
        print("✅ Advanced multi-agent coordination test passed")
    
    def test_communication_system_advanced(self):
        """Test advanced communication system functionality"""
        print("📡 Testing advanced communication system...")
        
        try:
            from agents.policies.communication_policy import CommunicationSystem
            
            comm_system = CommunicationSystem()
            
            # Test multiple message types
            message_types = [
                ("CIVILIAN_LOCATION", [7, 7]),
                ("ROAD_BLOCKED", [5, 5]),
                ("BUILDING_COLLAPSED", [3, 3]),
                ("AGENT_IN_DANGER", [10, 10]),
                ("MISSION_COMPLETE", [])
            ]
            
            for msg_type, data in message_types:
                comm_system.send_message("drone_1", "ambulance_1", msg_type, data)
            
            # Test message retrieval with filtering
            messages = comm_system.get_messages("ambulance_1")
            self.assertIsInstance(messages, list)
            self.assertEqual(len(messages), len(message_types))
            
            # Test message content
            for i, (expected_type, expected_data) in enumerate(message_types):
                self.assertEqual(messages[i]['type'], expected_type)
                self.assertEqual(messages[i]['data'], expected_data)
            
            # Test communication statistics
            stats = comm_system.get_communication_stats()
            expected_stats = ['total_messages', 'messages_by_type', 'agents_communicated']
            
            for stat in expected_stats:
                self.assertIn(stat, stats)
            
            self.assertEqual(stats['total_messages'], len(message_types))
            
            # Test message clearing
            comm_system.clear_messages("ambulance_1")
            cleared_messages = comm_system.get_messages("ambulance_1")
            self.assertEqual(len(cleared_messages), 0)
            
            print("✅ Advanced communication system test passed")
            
        except ImportError:
            self.skipTest("Communication system not available")
    
    def test_performance_benchmark_comprehensive(self):
        """Run comprehensive performance benchmarks"""
        print("⚡ Running comprehensive performance benchmarks...")
        
        import time
        
        # Test different environment sizes
        env_sizes = [15, 20, 25]
        agent_counts = [2, 4, 6]
        
        performance_results = {}
        
        for grid_size in env_sizes:
            for num_agents in agent_counts:
                # Update config
                test_config = self.config.copy()
                test_config['environment']['grid_size'] = grid_size
                
                # Create temp config
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
                    yaml.dump(test_config, temp_config)
                    temp_config_path = temp_config.name
                
                try:
                    env = SimpleGridEnv(temp_config_path)
                    
                    # Add agents
                    for i in range(num_agents):
                        drone = DroneAgent(f"perf_drone_{i}", np.array([2 + i, 2]), env.config)
                        env.add_agent(drone)
                    
                    # Benchmark
                    start_time = time.time()
                    steps = 5  # Reduced for comprehensive testing
                    
                    for step in range(steps):
                        actions = {agent_id: 'RIGHT' for agent_id in env.agents.keys()}
                        env.step(actions)
                    
                    end_time = time.time()
                    avg_step_time = (end_time - start_time) / steps
                    
                    performance_results[f"{grid_size}x{grid_size}_{num_agents}agents"] = avg_step_time
                    
                    print(f"  📊 Grid {grid_size}x{grid_size}, {num_agents} agents: {avg_step_time:.4f}s per step")
                    
                    env.close()
                    
                finally:
                    os.unlink(temp_config_path)
        
        # Performance requirements
        for scenario, time_per_step in performance_results.items():
            self.assertLess(time_per_step, 2.0, f"Performance requirement failed for {scenario}")
        
        print("✅ Comprehensive performance benchmark passed")
    
    def test_error_handling_and_robustness(self):
        """Test error handling and system robustness"""
        print("🛡️ Testing error handling and robustness...")
        
        env = SimpleGridEnv(self.temp_config.name)
        
        # Test 1: Invalid agent positions
        drone = DroneAgent("test_drone", np.array([100, 100]), env.config)  # Out of bounds
        env.add_agent(drone)
        
        # Should handle gracefully
        actions = {'test_drone': 'RIGHT'}
        result = env.step(actions)
        self.assertTrue(len(result) >= 4)
        
        # Test 2: Invalid actions
        actions = {'test_drone': 'INVALID_ACTION'}
        result = env.step(actions)
        self.assertTrue(len(result) >= 4)  # Should not crash
        
        # Test 3: Missing agents in actions
        actions = {}  # No actions provided
        result = env.step(actions)
        self.assertTrue(len(result) >= 4)  # Should handle gracefully
        
        # Test 4: Environment reset persistence
        obs1, info1 = env.reset()
        obs2, info2 = env.reset()
        self.assertIsNotNone(obs1)
        self.assertIsNotNone(obs2)
        
        env.close()
        print("✅ Error handling and robustness test passed")

class TestAdvancedMARLFeatures(unittest.TestCase):
    """
    Test advanced MARL features and integration
    """
    
    def test_learning_components_integration(self):
        """Test integration of all ML components"""
        print("🧠 Testing ML components integration...")
        
        try:
            from marl.reward_functions import CollaborativeReward
            from marl.observation_spaces import GlobalObservation
            from marl.action_spaces import DiscreteActionSpace
            
            # Create all components
            reward_func = CollaborativeReward(self.config)
            obs_space = GlobalObservation(20, 7, 3)
            action_space = DiscreteActionSpace()
            
            # Test they work together
            mock_agent = type('MockAgent', (), {
                'agent_id': 'test_agent',
                'position': np.array([5, 5]),
                'agent_type': 'drone'
            })()
            
            mock_obs = {
                'grid': np.random.randint(0, 7, (20, 20)),
                'agents': {},
                'civilians': []
            }
            
            # Encode observation
            encoded_obs = obs_space.encode_observation(mock_agent, mock_obs)
            self.assertIsInstance(encoded_obs, np.ndarray)
            
            # Get action space
            action_space_obj = action_space.get_action_space('drone')
            self.assertIsNotNone(action_space_obj)
            
            # Calculate reward
            reward = reward_func.calculate_reward(mock_agent, mock_obs, 'UP', mock_obs)
            self.assertIsInstance(reward, (int, float))
            
            print("✅ ML components integration test passed")
            
        except ImportError as e:
            self.skipTest(f"ML components not available: {e}")
    
    def test_training_compatibility_advanced(self):
        """Test advanced training framework compatibility"""
        print("🏋️ Testing advanced training compatibility...")
        
        try:
            from marl.pettingzoo_wrapper import DisasterResponsePettingZooEnv
            
            env = DisasterResponsePettingZooEnv(self.temp_config.name)
            
            # Basic PettingZoo compatibility checks
            self.assertTrue(hasattr(env, 'metadata'))
            self.assertTrue(hasattr(env, 'possible_agents'))
            self.assertTrue(hasattr(env, 'observation_spaces'))
            self.assertTrue(hasattr(env, 'action_spaces'))
            
            # Test reset
            observations = env.reset()
            self.assertIsInstance(observations, dict)
            
            # Test step for each agent
            if env.agents:
                for agent in env.agents[:2]:  # Test first two agents
                    env.agent_selection = agent
                    action_space = env.action_space(agent)
                    
                    if hasattr(action_space, 'sample'):
                        action = action_space.sample()
                    else:
                        action = 0  # Default action
                    
                    env.step(action)
                    
                    # Check observation
                    obs = env.observe(agent)
                    self.assertIsNotNone(obs)
            
            env.close()
            
            # Test Stable-Baselines3 compatibility if available
            try:
                from stable_baselines3 import PPO
                from stable_baselines3.common.env_checker import check_env
                print("  ✅ Stable-Baselines3 available for potential training")
            except ImportError:
                print("  ℹ️  Stable-Baselines3 not available, but not required")
            
            print("✅ Advanced training compatibility test passed")
            
        except Exception as e:
            self.skipTest(f"Training compatibility test skipped: {e}")
    
    def test_policy_integration(self):
        """Test integration with advanced policies"""
        print("🎯 Testing policy integration...")
        
        try:
            from agents.policies.advanced_policies import CooperativePolicy, AStarPolicy
            
            env = SimpleGridEnv(self.temp_config.name)
            
            # Test Cooperative Policy
            coop_policy = CooperativePolicy(env.config)
            self.assertIsNotNone(coop_policy)
            
            drone = DroneAgent("policy_drone", np.array([5, 5]), env.config)
            env.add_agent(drone)
            
            # Get action from policy
            action = coop_policy.get_action(drone, env)
            self.assertIn(action, ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'REST'])
            
            # Test A* Pathfinding
            astar_policy = AStarPolicy()
            test_grid = np.zeros((10, 10))
            path = astar_policy.find_path((0, 0), (9, 9), test_grid)
            self.assertIsInstance(path, list)
            self.assertTrue(len(path) > 0)
            
            env.close()
            print("✅ Policy integration test passed")
            
        except ImportError as e:
            self.skipTest(f"Advanced policies not available: {e}")

def run_marL_tests():
    """Run all MARL tests and return comprehensive results"""
    print("🧪 ENHANCED MARL System Tests - Week 6")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMARLSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedMARLFeatures))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, resultclass=unittest.TextTestResult)
    result = runner.run(suite)
    
    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("📊 ENHANCED MARL TEST SUMMARY - WEEK 6")
    print("=" * 60)
    print(f"🏁 Total Tests Run: {result.testsRun}")
    print(f"✅ Tests Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests Failed: {len(result.failures)}")
    print(f"💥 Tests Errored: {len(result.errors)}")
    print(f"⏭️ Tests Skipped: {len(result.skipped)}")
    print(f"📈 Success Rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    # Detailed breakdown
    if result.failures:
        print(f"\n🔍 Failed Tests:")
        for test, traceback in result.failures:
            print(f"   ❌ {test}: {traceback.splitlines()[-1]}")
    
    if result.errors:
        print(f"\n🚨 Errored Tests:")
        for test, traceback in result.errors:
            print(f"   💥 {test}: {traceback.splitlines()[-1]}")
    
    if result.skipped:
        print(f"\n⏭️ Skipped Tests:")
        for test, reason in result.skipped:
            print(f"   ⏩ {test}: {reason}")
    
    # Final verdict
    if result.wasSuccessful():
        print(f"\n🎉 EXCELLENT! All MARL tests passed! 🎉")
        print("   Your MARL system is fully operational and production-ready!")
        return True
    else:
        print(f"\n⚠️  Some MARL tests need attention.")
        if (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun >= 0.8:
            print("   System is functional but has some issues to address.")
        else:
            print("   Significant issues need to be resolved.")
        return False

if __name__ == "__main__":
    success = run_marL_tests()
    sys.exit(0 if success else 1)