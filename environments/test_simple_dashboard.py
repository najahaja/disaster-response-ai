import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Test Dashboard", layout="wide")

st.title("🚨 Disaster Response AI - Test Version")

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'sim_started' not in st.session_state:
    st.session_state.sim_started = False
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = {'steps': [], 'start_time': None}

# Simple authentication
if not st.session_state.logged_in:
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if (username == "admin" and password == "Admin@123") or \
           (username == "viewer" and password == "Viewer@123"):
            st.session_state.logged_in = True
            st.session_state.user_role = "admin" if username == "admin" else "viewer"
            st.success(f"✅ Logged in as {st.session_state.user_role}")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# Main dashboard - only reached if logged_in is True
st.success(f"✅ Logged in as {st.session_state.user_role}")

# Sidebar
with st.sidebar:
    st.title("🎮 Controls")
    
    # User info
    st.write(f"**Role:** {st.session_state.user_role}")
    
    # Simulation controls
    if st.button("🚀 Start Simulation", use_container_width=True):
        st.session_state.sim_started = True
        st.session_state.sim_data = {'steps': [], 'start_time': datetime.now()}
        st.success("✅ Simulation started!")
        st.rerun()
    
    if st.button("⏹️ Stop Simulation", use_container_width=True, 
                disabled=not st.session_state.sim_started):
        st.session_state.sim_started = False
        st.info("⏹️ Simulation stopped")
        st.rerun()
    
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.sim_started = False
        st.session_state.sim_data = {'steps': [], 'start_time': None}
        st.rerun()
    
    st.divider()
    
    if st.button("🚪 Logout", use_container_width=True):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main content
if st.session_state.sim_started:
    st.subheader("📊 Simulation Running")
    
    # Show metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", "🟢 Running")
    with col2:
        if st.session_state.sim_data['start_time']:
            elapsed = datetime.now() - st.session_state.sim_data['start_time']
            st.metric("Time", f"{elapsed.seconds}s")
        else:
            st.metric("Time", "0s")
    with col3:
        st.metric("Agents Active", "5")
    with col4:
        st.metric("Rescues", "12")
    
    # Simple visualization
    import plotly.graph_objects as go
    import numpy as np
    
    # Create a grid
    grid = np.random.rand(10, 10)
    fig = go.Figure(data=go.Heatmap(z=grid, colorscale='Viridis'))
    fig.update_layout(title="Live Simulation Grid", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add a step to simulate progress
    if st.button("⏭️ Next Step", type="primary"):
        # Add a step to simulation data
        step_data = {
            'step': len(st.session_state.sim_data['steps']) + 1,
            'timestamp': datetime.now(),
            'rescues': np.random.randint(0, 3)
        }
        st.session_state.sim_data['steps'].append(step_data)
        st.rerun()
    
    # Show steps
    if st.session_state.sim_data['steps']:
        st.subheader("📋 Simulation Steps")
        import pandas as pd
        df = pd.DataFrame(st.session_state.sim_data['steps'])
        st.dataframe(df, use_container_width=True)
    
    # Progress bar
    progress = min(1.0, len(st.session_state.sim_data['steps']) / 20)
    st.progress(progress, text=f"Simulation Progress: {progress*100:.0f}%")
    
else:
    st.info("👋 Ready to start simulation")
    st.write("Click '🚀 Start Simulation' in the sidebar to begin")
    
    # Show sample data when not running
    st.subheader("📈 Sample Performance Data")
    import plotly.express as px
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'Time Step': range(20),
        'Rescues': [max(0, (x-2)*3) for x in range(20)],
        'Efficiency': [min(100, x*8 + 10) for x in range(20)]
    })
    
    fig = px.line(sample_data, x='Time Step', y=['Rescues', 'Efficiency'],
                 title="Expected Performance Trends")
    st.plotly_chart(fig, use_container_width=True)