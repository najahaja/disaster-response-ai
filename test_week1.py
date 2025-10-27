#!/usr/bin/env python3
"""
Comprehensive test script for Week 1 functionality
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.simple_grid_env import SimpleGridEnv
from agents.drone_agent import DroneAgent
from agents.ambulance_agent import AmbulanceAgent
from agents.rescue_team_agent import RescueTeamAgent

def test_environment_creation():
    """Test environment creation and basic functionality"""
    print("🧪 Testing Environment Creation...")
    
    env = SimpleGridEnv("config.yaml")
    
    # Test grid initialization
    assert env.grid is not None, "Grid should be initialized"
    assert env.grid.shape == (env.grid_size, env.grid_size), "Grid shape should match config"
    
    # Test cell types
    unique_cells = np.unique(env.grid)
    expected_cells = list(env.cell_types.values())
    for cell in unique_cells:
        assert cell in expected_cells, f"Unexpected cell type: {cell}"
    
    print("✅ Environment creation test passed!")

def test_agent_creation():
    """Test agent creation and basic functionality"""
    print("🧪 Testing Agent Creation...")
    
    env = SimpleGridEnv("config.yaml")
    
    # Create agents
    drone = DroneAgent("test_drone", [5, 5], env.config)
    ambulance = AmbulanceAgent("test_ambulance", [10, 10], env.config)
    rescue_team = RescueTeamAgent("test_rescue", [8, 8], env.config)
    
    # Test agent properties
    assert drone.agent_type == 'drone', "Drone type should be 'drone'"
    assert ambulance.capacity == 5, "Ambulance capacity should be 5"
    assert rescue_team.speed == 1, "Rescue team speed should be 1"
    
    # Test agent movement
    initial_pos = drone.position.copy()
    success = drone.move('RIGHT', env.grid)
    assert success, "Movement should be successful"
    assert not np.array_equal(drone.position, initial_pos), "Position should change after movement"
    
    print("✅ Agent creation test passed!")

def test_disaster_generation():
    """Test disaster scenario generation"""
    print("🧪 Testing Disaster Generation...")
    
    env = SimpleGridEnv("config.yaml")
    
    # Count initial buildings and roads
    initial_buildings = np.sum(env.grid == env.cell_types['BUILDING'])
    initial_roads = np.sum(env.grid == env.cell_types['ROAD'])
    
    # Trigger disaster
    env.trigger_disaster()
    
    # Check disaster effects
    collapsed_count = np.sum(env.grid == env.cell_types['COLLAPSED'])
    blocked_count = np.sum(env.grid == env.cell_types['BLOCKED'])
    
    assert collapsed_count > 0, "Should have collapsed buildings"
    assert blocked_count > 0, "Should have blocked roads"
    assert len(env.civilians) > 0, "Should have spawned civilians"
    
    print("✅ Disaster generation test passed!")

def test_environment_step():
    """Test environment stepping functionality"""
    print("🧪 Testing Environment Step...")
    
    env = SimpleGridEnv("config.yaml")
    drone = DroneAgent("test_drone", [5, 5], env.config)
    env.add_agent(drone)
    env.trigger_disaster()
    
    # Take a step
    actions = {"test_drone": "RIGHT"}
    observation, rewards, done, info = env.step(actions)
    
    # Check results
    assert env.step_count == 1, "Step count should increment"
    assert "test_drone" in rewards, "Should have reward for test_drone"
    assert not done, "Should not be done after one step"
    assert "grid" in observation, "Observation should include grid"
    
    print("✅ Environment step test passed!")

def run_all_tests():
    """Run all Week 1 tests"""
    print("=" * 60)
    print("           WEEK 1 COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    try:
        test_environment_creation()
        test_agent_creation() 
        test_disaster_generation()
        test_environment_step()
        
        print("\n" + "=" * 60)
        print("🎉 ALL WEEK 1 TESTS PASSED! 🎉")
        print("✅ Environment: Working")
        print("✅ Agents: Functional") 
        print("✅ Disaster System: Operational")
        print("✅ Simulation: Runnable")
        print("=" * 60)
        print("🚀 Ready to proceed to Week 2: MARL Integration!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()