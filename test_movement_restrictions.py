"""
Test script to verify agent movement restrictions
- Drones CAN fly over buildings
- Ambulances and Rescue Teams CANNOT move through buildings
"""

import sys
import numpy as np
sys.path.append('.')

from environments.simple_grid_env import SimpleGridEnv

def test_movement_restrictions():
    print("=" * 60)
    print("Testing Agent Movement Restrictions")
    print("=" * 60)
    
    # Create environment with 1 of each agent type
    env = SimpleGridEnv(
        grid_size=10,
        n_drones=1,
        n_ambulances=1,
        n_rescue_teams=1,
        spawn_civilians=False  # No civilians for this test
    )
    
    # Get agents
    drone = None
    ambulance = None
    rescue_team = None
    
    for agent_id, agent in env.agents.items():
        if agent.agent_type == 'drone':
            drone = agent
        elif agent.agent_type == 'ambulance':
            ambulance = agent
        elif agent.agent_type == 'rescue_team':
            rescue_team = agent
    
    print(f"\n✅ Environment created with {len(env.agents)} agents")
    print(f"   - Drone at {drone.position}")
    print(f"   - Ambulance at {ambulance.position}")
    print(f"   - Rescue Team at {rescue_team.position}")
    
    # Find a building cell
    building_type = env.cell_types['BUILDING']
    building_pos = None
    for y in range(env.grid_size):
        for x in range(env.grid_size):
            if env.grid[y, x] == building_type:
                building_pos = np.array([x, y])
                break
        if building_pos is not None:
            break
    
    print(f"\n📍 Found building at {building_pos}")
    
    # Test 1: Try to move drone to building
    print("\n" + "=" * 60)
    print("TEST 1: Drone Movement (Should SUCCEED)")
    print("=" * 60)
    drone.position = building_pos - np.array([1, 0])  # Position next to building
    print(f"Drone starting position: {drone.position}")
    print(f"Attempting to move drone RIGHT into building at {building_pos}...")
    
    success = drone.move('RIGHT', env.grid)
    if success:
        print(f"✅ SUCCESS! Drone moved to {drone.position} (can fly over buildings)")
    else:
        print(f"❌ FAILED! Drone could not move (unexpected)")
    
    # Test 2: Try to move ambulance to building
    print("\n" + "=" * 60)
    print("TEST 2: Ambulance Movement (Should FAIL)")
    print("=" * 60)
    ambulance.position = building_pos - np.array([1, 0])  # Position next to building
    print(f"Ambulance starting position: {ambulance.position}")
    print(f"Attempting to move ambulance RIGHT into building at {building_pos}...")
    
    success = ambulance.move('RIGHT', env.grid)
    if not success:
        print(f"✅ SUCCESS! Ambulance blocked from building (ground vehicle restriction working)")
    else:
        print(f"❌ FAILED! Ambulance moved through building (bug!)")
    
    # Test 3: Try to move rescue team to building
    print("\n" + "=" * 60)
    print("TEST 3: Rescue Team Movement (Should FAIL)")
    print("=" * 60)
    rescue_team.position = building_pos - np.array([1, 0])  # Position next to building
    print(f"Rescue Team starting position: {rescue_team.position}")
    print(f"Attempting to move rescue team RIGHT into building at {building_pos}...")
    
    success = rescue_team.move('RIGHT', env.grid)
    if not success:
        print(f"✅ SUCCESS! Rescue Team blocked from building (ground vehicle restriction working)")
    else:
        print(f"❌ FAILED! Rescue Team moved through building (bug!)")
    
    # Test 4: Verify ground vehicles CAN move on roads
    print("\n" + "=" * 60)
    print("TEST 4: Ground Vehicle on Road (Should SUCCEED)")
    print("=" * 60)
    
    # Find a road cell
    road_type = env.cell_types['ROAD']
    road_pos = None
    for y in range(env.grid_size):
        for x in range(env.grid_size):
            if env.grid[y, x] == road_type:
                road_pos = np.array([x, y])
                break
        if road_pos is not None:
            break
    
    ambulance.position = road_pos - np.array([1, 0])  # Position next to road
    print(f"Ambulance starting position: {ambulance.position}")
    print(f"Attempting to move ambulance RIGHT onto road at {road_pos}...")
    
    success = ambulance.move('RIGHT', env.grid)
    if success:
        print(f"✅ SUCCESS! Ambulance moved to road at {ambulance.position}")
    else:
        print(f"❌ FAILED! Ambulance could not move to road (unexpected)")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    
    env.close()

if __name__ == "__main__":
    test_movement_restrictions()
