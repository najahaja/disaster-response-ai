# Multi-Agent Coordination System

## Overview

The disaster response AI now features a **sophisticated communication and coordination system** that enables agents to work together effectively, especially for rescuing civilians trapped in buildings.

## The Problem

**Civilians in buildings cannot be reached by ground vehicles:**

- Ground vehicles (ambulances, rescue teams) cannot move through buildings
- Drones can fly over buildings but have limited rescue capacity
- **Solution**: Drones scout and share coordinates with ground teams

## How It Works

### 1. Communication Hub

Central coordination system (`agents/communication_hub.py`) that manages:

- **Discovered civilian locations**
- **Rescue requests and assignments**
- **Inter-agent messaging**
- **Coordination statistics**

### 2. Agent Roles

#### 🚁 **Drones (Scouts)**

- **Fly over buildings** to find civilians
- **Scout range**: 3 cells around drone
- **Automatically report** civilian locations to communication hub
- **Broadcast discovery** messages to all agents
- Can rescue civilians but have limited capacity

#### 🚑 **Ambulances (Transport)**

- **Ground vehicle** - restricted to roads and open spaces
- **Receive coordinates** from communication hub
- **Navigate to nearest** unassigned civilian
- **High capacity** for transporting multiple civilians
- Must reach civilians via roads

#### 🚒 **Rescue Teams (Extraction)**

- **Ground vehicle** - restricted to roads and open spaces
- **Specialized** in complex rescue operations
- **Receive coordinates** from communication hub
- Can access **collapsed buildings** to extract civilians
- Navigate to rescue locations via roads

### 3. Coordination Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    COORDINATION WORKFLOW                     │
└─────────────────────────────────────────────────────────────┘

1. DISCOVERY PHASE
   ┌──────────┐
   │  Drone   │ ──► Flies over buildings
   └──────────┘
        │
        ▼
   Finds civilian in building
        │
        ▼
   ┌─────────────────────┐
   │ Communication Hub   │ ◄── Reports location
   └─────────────────────┘
        │
        ▼
   Broadcasts to all agents

2. ASSIGNMENT PHASE
   ┌─────────────────────┐
   │ Communication Hub   │
   └─────────────────────┘
        │
        ├──► Ambulance queries: "Nearest civilian?"
        │
        └──► Hub assigns: "Go to [x, y]"

3. RESCUE PHASE
   ┌──────────────┐
   │  Ambulance   │ ──► Navigates via roads
   └──────────────┘
        │
        ▼
   Reaches adjacent cell to building
        │
        ▼
   Rescues civilian
        │
        ▼
   ┌─────────────────────┐
   │ Communication Hub   │ ◄── Marks as rescued
   └─────────────────────┘
```

### 4. Key Features

#### **Automatic Discovery**

```python
# Drones automatically scout every step
def _drone_scout_and_report(drone):
    for civilian in civilians_in_range:
        if not discovered:
            comm_hub.report_civilian_location(
                civilian_id, position, drone.agent_id
            )
```

#### **Smart Assignment**

```python
# Ground vehicles get nearest unassigned civilian
nearest_civ = comm_hub.get_nearest_unassigned_civilian(
    agent_position=ambulance.position,
    agent_type='ambulance'
)
comm_hub.assign_civilian_to_agent(civilian_id, ambulance.agent_id)
```

#### **Status Tracking**

```python
# Real-time coordination statistics
stats = comm_hub.get_coordination_stats()
# Returns:
# - total_discovered
# - rescued
# - in_progress
# - unassigned
# - pending_requests
```

## API Reference

### CommunicationHub Methods

#### `report_civilian_location(civilian_id, position, discovered_by_agent)`

Agent reports a discovered civilian

- **Returns**: `True` if new discovery, `False` if already known

#### `get_nearest_unassigned_civilian(agent_position, agent_type=None)`

Find nearest civilian that hasn't been assigned

- **Returns**: Dictionary with civilian info or `None`

#### `assign_civilian_to_agent(civilian_id, agent_id)`

Assign a civilian rescue to a specific agent

- **Returns**: `True` if successful

#### `mark_civilian_rescued(civilian_id, rescued_by_agent)`

Mark a civilian as successfully rescued

- **Returns**: `True` if successful

#### `broadcast_message(sender_id, message_type, content)`

Broadcast a message to all agents

- **Message types**: `'civilian_found'`, `'civilian_rescued'`, `'need_assistance'`

#### `get_coordination_stats()`

Get statistics about coordination

- **Returns**: Dictionary with counts and status

### Environment Methods

#### `get_coordination_info()`

Get current coordination status

```python
info = env.get_coordination_info()
# Returns:
# {
#     'discovered_civilians': {...},
#     'pending_requests': [...],
#     'stats': {...}
# }
```

## Usage Examples

### Example 1: Basic Coordination

```python
from environments.simple_grid_env import SimpleGridEnv

# Create environment with coordination enabled
env = SimpleGridEnv(
    grid_size=36,
    n_drones=2,
    n_ambulances=2,
    n_rescue_teams=1,
    spawn_civilians=True,
    n_civilians=10
)

# Run simulation
for step in range(100):
    action = agent.select_action(obs)
    obs, reward, done, truncated, info = env.step(action)

    # Check coordination status
    coord_info = env.get_coordination_info()
    print(f"Discovered: {coord_info['stats']['total_discovered']}")
    print(f"Rescued: {coord_info['stats']['rescued']}")
```

### Example 2: Manual Coordination

```python
# Access communication hub directly
comm_hub = env.comm_hub

# Report a civilian (usually done automatically by drones)
comm_hub.report_civilian_location(
    civilian_id="civ_1",
    position=[15, 20],
    discovered_by_agent="drone_0"
)

# Get nearest civilian for an agent
nearest = comm_hub.get_nearest_unassigned_civilian(
    agent_position=[10, 10],
    agent_type='ambulance'
)

# Assign to agent
if nearest:
    comm_hub.assign_civilian_to_agent(
        civilian_id=nearest['civilian_id'],
        agent_id="ambulance_0"
    )
```

## Benefits

### For Training

- **Realistic scenarios**: Agents must coordinate like real rescue teams
- **Emergent behavior**: AI learns to scout, communicate, and coordinate
- **Scalable**: Works with any number of agents

### For Simulation

- **Accurate modeling**: Reflects real disaster response operations
- **Performance tracking**: Monitor coordination effectiveness
- **Debugging**: See exactly what each agent knows and is doing

## Testing

Run the coordination demonstration:

```bash
python demo_coordination.py
```

Run movement restrictions test:

```bash
python test_movement_restrictions.py
```

## Integration with Training

The coordination system is **automatically active** during training:

```python
# In config.json
{
    "n_drones": 2,
    "n_ambulances": 2,
    "n_rescue_teams": 1,
    "spawn_civilians": true,
    "n_civilians": 10
}
```

The AI will learn to:

1. **Scout efficiently** with drones
2. **Navigate around buildings** with ground vehicles
3. **Coordinate rescues** using shared information
4. **Prioritize** based on distance and urgency

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SimpleGridEnv                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │              Communication Hub                      │     │
│  │  - Discovered civilians                            │     │
│  │  - Rescue requests                                 │     │
│  │  - Agent messages                                  │     │
│  │  - Coordination stats                              │     │
│  └────────────────────────────────────────────────────┘     │
│         ▲                    ▲                    ▲          │
│         │                    │                    │          │
│    ┌────┴────┐         ┌────┴────┐         ┌────┴────┐     │
│    │  Drone  │         │Ambulance│         │ Rescue  │     │
│    │         │         │         │         │  Team   │     │
│    │ Scout & │         │ Receive │         │ Receive │     │
│    │ Report  │         │ Coords  │         │ Coords  │     │
│    └─────────┘         └─────────┘         └─────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Future Enhancements

Potential improvements:

- **Priority levels** for urgent rescues
- **Resource allocation** (fuel, medical supplies)
- **Dynamic reassignment** if agent fails
- **Multi-hop communication** for large areas
- **Pathfinding integration** for optimal routes

---

**The coordination system makes your disaster response AI realistic, scalable, and effective!** 🚁🚑🚒
