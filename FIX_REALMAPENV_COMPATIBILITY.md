# Fix: RealMapEnv Compatibility

## Issue

```
Failed to start simulation: 'RealMapEnv' object has no attribute 'n_drones'
```

## Root Cause

When we added agent configuration parameters (`n_drones`, `n_ambulances`, `n_rescue_teams`, `spawn_civilians`, `n_civilians`) to `SimpleGridEnv`, we didn't update `RealMapEnv` which inherits from it.

The dashboard was trying to access these attributes on `RealMapEnv` instances, causing the error.

## Solution

Updated `RealMapEnv.__init__()` to accept the same parameters as `SimpleGridEnv`:

```python
def __init__(self, location_name=None, config_path="config.yaml",
             n_drones=1, n_ambulances=1, n_rescue_teams=1,
             spawn_civilians=True, n_civilians=None):
    # Store agent configuration
    self.n_drones = n_drones
    self.n_ambulances = n_ambulances
    self.n_rescue_teams = n_rescue_teams
    self.spawn_civilians = spawn_civilians
    self.n_civilians = n_civilians
    # ... rest of initialization
```

## Files Modified

- `environments/real_map_env.py` - Added agent configuration parameters

## Testing

The dashboard should now work correctly with both `SimpleGridEnv` and `RealMapEnv`:

```bash
# Run dashboard
python run_dashboard.py
```

## Compatibility

Both environments now support:

- Configurable number of drones, ambulances, and rescue teams
- Optional civilian spawning
- Specific civilian counts
- Communication hub for coordination

This ensures consistent behavior whether using:

- Simple grid environment (training)
- Real map environment (dashboard/testing)
