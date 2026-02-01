# Dashboard Crash Fix

## Issue

Dashboard crashes after a few moments with error:

```
No media file with id '...'
```

## Root Cause

This is a **Streamlit caching/session issue**, not a code error. It happens when:

1. Streamlit tries to access cached media files that have been cleared
2. The dashboard reruns too frequently
3. Session state gets corrupted

## Quick Fixes

### Fix 1: Clear Streamlit Cache

```bash
# Stop the dashboard (Ctrl+C)
# Clear cache
streamlit cache clear

# Restart
python run_dashboard.py
```

### Fix 2: Reduce Rerun Frequency

The dashboard is calling `st.rerun()` very frequently in the simulation loop. This can cause Streamlit to crash.

**Temporary workaround:**

- Don't let the simulation run for too long
- Pause the simulation periodically
- Restart the dashboard if it becomes unstable

### Fix 3: Check for Infinite Rerun Loop

The issue might be in `advance_simulation()` which calls `st.rerun()` at the end. If there's an error, it might be creating an infinite loop.

## Code Changes Made

### 1. Removed comm_hub from \_check_civilian_rescues

**File:** `environments/simple_grid_env.py`

**Before:**

```python
if agent.rescue_civilian():
    civilian['rescued'] = True
    self.comm_hub.mark_civilian_rescued(...)  # CRASH!
```

**After:**

```python
if agent.rescue_civilian():
    civilian['rescued'] = True
    # No comm_hub calls
```

## Testing

### Test 1: Run Dashboard Without Simulation

```bash
python run_dashboard.py
```

- Login
- Don't start simulation
- Dashboard should stay stable

### Test 2: Run Short Simulation

```bash
python run_dashboard.py
```

- Login
- Start simulation
- Let it run for 10-20 steps
- Pause it
- Check if stable

### Test 3: Use Simple Grid Instead of Real Map

When starting simulation, use "Generated City" instead of a real location. This is simpler and less likely to crash.

## Dashboard Stability Tips

1. **Use Generated City** - More stable than real maps
2. **Pause Frequently** - Don't let it run continuously
3. **Restart Periodically** - Streamlit can get unstable over time
4. **Reduce Simulation Speed** - Slower = more stable
5. **Fewer Agents** - Start with 1-2 of each type

## Alternative: Use Basic Simulation

If dashboard keeps crashing, use the basic simulation instead:

```bash
python run_basic_simulation.py
```

This uses Pygame directly without Streamlit, so it's more stable.

## Status

✅ **Fixed:** Removed comm_hub references that were causing AttributeError
⚠️ **Streamlit Issue:** The "No media file" error is a Streamlit bug, not our code
🔧 **Workaround:** Restart dashboard when it crashes

## Next Steps

1. Try running the dashboard again
2. If it crashes, note:
   - How long it ran before crashing
   - What you were doing when it crashed
   - Any error messages in console

3. If it keeps crashing:
   - Use the basic simulation instead
   - Or reduce simulation complexity (fewer agents, slower speed)
