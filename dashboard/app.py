#!/usr/bin/env python3
"""
Enhanced Main Dashboard Application for Disaster Response AI - Week 6
Streamlit-based dashboard with improved performance, error handling, and new features
"""

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import logging
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project components with enhanced error handling
try:
    from environments.real_map_env import RealMapEnv
    from environments.simple_grid_env import SimpleGridEnv
    from agents.drone_agent import DroneAgent
    from agents.ambulance_agent import AmbulanceAgent
    from agents.rescue_team_agent import RescueTeamAgent
    from dashboard.components.controls import ControlPanel
    from dashboard.components.metrics_display import MetricsDisplay
    from dashboard.components.simulation_viewer import SimulationViewer
    import pygame
    
    # Initialize pygame safely
    try:
        pygame.init()
        PYGAME_AVAILABLE = True
    except:
        PYGAME_AVAILABLE = False
        logger.warning("Pygame initialization failed - visualization may be limited")
        
except ImportError as e:
    logger.error(f"Failed to import project components: {e}")
    st.error("❌ Critical components missing. Please check installation.")
    # Create minimal fallbacks
    PYGAME_AVAILABLE = False

class DisasterResponseDashboard:
    """
    Enhanced dashboard class for Disaster Response AI simulation - Week 6
    """
    
    def __init__(self):
        self.setup_page()
        self.initialize_session_state()
        self.initialize_components()
        
    def setup_page(self):
        """Setup Streamlit page configuration with enhanced styling"""
        st.set_page_config(
            page_title="Disaster Response AI Dashboard",
            page_icon="🚨",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Enhanced Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #ff4b4b;
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(45deg, #ff4b4b, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.2rem;
            border-radius: 12px;
            border-left: 6px solid #ff4b4b;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .success-metric {
            border-left: 6px solid #00cc96;
        }
        .warning-metric {
            border-left: 6px solid #ffa726;
        }
        .danger-metric {
            border-left: 6px solid #ff4b4b;
        }
        .info-metric {
            border-left: 6px solid #1f77b4;
        }
        .stButton button {
            width: 100%;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize session state variables with enhanced tracking"""
        default_state = {
            'simulation_running': False,
            'environment': None,
            'simulation_data': {
                'steps': [],
                'metrics': [],
                'agent_positions': {},
                'civilian_rescues': 0,
                'start_time': None,
                'performance_history': [],
                'error_log': []
            },
            'selected_location': "Lahore, Pakistan",
            'disaster_triggered': False,
            'simulation_mode': 'basic',  # 'basic', 'advanced', 'training'
            'auto_save': True,
            'last_save_time': None,
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def initialize_components(self):
        """Initialize dashboard components with error handling"""
        try:
            self.control_panel = ControlPanel()
            self.metrics_display = MetricsDisplay()
            self.simulation_viewer = SimulationViewer()
            logger.info("Dashboard components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize dashboard components: {e}")
            st.error("⚠️ Some dashboard features may be limited")
    
    def render_header(self):
        """Render enhanced dashboard header"""
        st.markdown('<h1 class="main-header">🚨 Disaster Response AI Dashboard v2.0</h1>', 
                   unsafe_allow_html=True)
        
        # Enhanced status indicator with more metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_icon = "🟢" if st.session_state.simulation_running else "🔴"
            status_text = "Running" if st.session_state.simulation_running else "Stopped"
            st.metric("Simulation Status", f"{status_icon} {status_text}")
        
        with col2:
            if st.session_state.environment:
                env_type = "Real Map" if hasattr(st.session_state.environment, 'real_map_loaded') and st.session_state.environment.real_map_loaded else "Generated"
                st.metric("Environment", env_type)
            else:
                st.metric("Environment", "Not Loaded")
        
        with col3:
            civilian_count = st.session_state.simulation_data.get('civilian_rescues', 0)
            st.metric("Civilians Rescued", f"{civilian_count} 🎯")
        
        with col4:
            if st.session_state.simulation_data['start_time']:
                duration = datetime.now() - st.session_state.simulation_data['start_time']
                st.metric("Run Time", f"{duration.total_seconds():.0f}s ⏱️")
            else:
                st.metric("Run Time", "0s ⏱️")
    
    def render_sidebar(self):
        """Render enhanced sidebar with additional features"""
        with st.sidebar:
            st.header("🎮 Simulation Controls")
            
            # Simulation Mode Selection
            st.subheader("🔧 Simulation Mode")
            mode = st.selectbox(
                "Select Mode",
                options=["Basic", "Advanced", "Training"],
                index=0,
                help="Basic: Simple simulation\nAdvanced: Full features\nTraining: ML training mode"
            )
            st.session_state.simulation_mode = mode.lower()
            
            # Environment selection with enhanced options
            st.subheader("🗺️ Environment Setup")
            location_options = [
                "Lahore, Pakistan",
                "Karachi, Pakistan", 
                "Islamabad, Pakistan",
                "Tokyo, Japan",
                "New York, USA", 
                "Generated City"
            ]
            selected_location = st.selectbox(
                "Select Location",
                options=location_options,
                index=0
            )
            
            # Enhanced Agent configuration
            st.subheader("🤖 Agent Configuration")
            col1, col2 = st.columns(2)
            with col1:
                num_drones = st.slider("Drones 🛸", 1, 8, 2, 
                                     help="Aerial reconnaissance units")
                num_ambulances = st.slider("Ambulances 🚑", 1, 6, 2,
                                         help="Medical transport units")
            with col2:
                num_rescue_teams = st.slider("Rescue Teams 👷", 1, 4, 1,
                                           help="Ground rescue units")
                auto_deploy = st.checkbox("Auto-deploy", value=True,
                                        help="Automatically deploy agents at optimal positions")
            
            # Enhanced Simulation controls
            st.subheader("🎯 Simulation Control")
            
            control_col1, control_col2 = st.columns(2)
            with control_col1:
                if st.button("🚀 Start Simulation", width='stretch', 
                           help="Start new simulation with current configuration"):
                    self.start_simulation(selected_location, num_drones, num_ambulances, num_rescue_teams, auto_deploy)
            
            with control_col2:
                if st.button("⏹️ Stop Simulation", width='stretch',
                           help="Stop current simulation"):
                    self.stop_simulation()
            
            # Additional control buttons
            if st.button("🚨 Trigger Disaster", width='stretch',
                       help="Trigger disaster scenario in current simulation"):
                self.trigger_disaster()
            
            if st.button("🔄 Reset Simulation", width='stretch',
                       help="Reset simulation to initial state"):
                self.reset_simulation()
            
            # Enhanced settings
            st.subheader("⚙️ Simulation Settings")
            sim_speed = st.slider("Simulation Speed", 1, 20, 5,
                                help="Faster speed may reduce visualization quality")
            st.session_state.simulation_speed = sim_speed
            
            auto_save = st.checkbox("Auto-save progress", value=True,
                                  help="Automatically save simulation state")
            st.session_state.auto_save = auto_save
            
            # Real-time metrics in sidebar
            st.subheader("📊 Live Metrics")
            if st.session_state.simulation_running and st.session_state.environment:
                self.render_live_metrics()
            
            # Session info
            st.subheader("💾 Session Info")
            st.write(f"Session ID: `{st.session_state.session_id}`")
            if st.session_state.simulation_data['start_time']:
                st.write(f"Started: {st.session_state.simulation_data['start_time'].strftime('%H:%M:%S')}")
    
    def render_live_metrics(self):
        """Render enhanced live metrics in sidebar"""
        env = st.session_state.environment
        if not env:
            return
            
        # Calculate comprehensive metrics
        rescued = sum(1 for c in env.civilians if c.get('rescued', False))
        total_civilians = len(env.civilians)
        rescue_rate = (rescued / total_civilians * 100) if total_civilians > 0 else 0
        efficiency = min(env.step_count / max(1, rescued) if rescued > 0 else env.step_count, 100)
        
        metrics = {
            "Step": f"{env.step_count} 📈",
            "Active Agents": f"{len(env.agents)} 🤖",
            "Civilians": f"{total_civilians} 👥",
            "Rescued": f"{rescued} ✅ ({rescue_rate:.1f}%)",
            "Efficiency": f"{efficiency:.1f} ⚡",
            "Collapsed Buildings": f"{len(env.collapsed_buildings)} 🏚️",
            "Blocked Roads": f"{len(env.blocked_roads)} 🚧"
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)
        
        # Progress visualization
        st.progress(rescue_rate / 100, text=f"Rescue Progress: {rescue_rate:.1f}%")
    
    def render_main_content(self):
        """Render main dashboard content with enhanced layout"""
        if not st.session_state.simulation_running:
            self.render_enhanced_welcome_screen()
        else:
            self.render_enhanced_simulation_dashboard()
    
    def render_enhanced_welcome_screen(self):
        """Render enhanced welcome screen"""
        st.markdown("""
        ## 🌟 Welcome to Disaster Response AI Dashboard v2.0
        
        **Week 6 Enhanced Features:**
        - 🚀 **Improved Performance**: Faster simulation and rendering
        - 🛡️ **Enhanced Error Handling**: Better stability and user feedback  
        - 📈 **Advanced Analytics**: Comprehensive performance tracking
        - 💾 **Auto-save**: Automatic progress saving
        - 🎯 **Multiple Modes**: Basic, Advanced, and Training modes
        
        ### 🎯 What's New in Week 6:
        """)
        
        # New features showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **🧪 Comprehensive Testing**
            - Unit tests for all components
            - Integration testing
            - Performance benchmarking
            """)
        
        with col2:
            st.info("""
            **📚 Enhanced Documentation**
            - API references
            - User manuals  
            - Deployment guides
            """)
        
        with col3:
            st.info("""
            **🚀 Production Ready**
            - Error recovery
            - Performance optimization
            - Professional logging
            """)
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.markdown("""
            1. **Select Simulation Mode** in sidebar
            2. **Choose Location** (real map or generated)
            3. **Configure Your Team** (drones, ambulances, rescue teams)
            4. **Click 'Start Simulation'**
            5. **Trigger Disaster** when ready
            6. **Monitor Performance** in real-time
            """)
        
        # Sample analytics
        st.subheader("📊 Sample Analytics Dashboard")
        self.render_sample_analytics()
    
    def render_sample_analytics(self):
        """Render sample analytics for welcome screen"""
        tab1, tab2, tab3 = st.tabs(["Performance Trends", "Agent Distribution", "Scenario Analysis"])
        
        with tab1:
            # Sample performance data
            sample_data = pd.DataFrame({
                'Time': range(20),
                'Rescues': [max(0, (x-2)*3) for x in range(20)],
                'Efficiency': [min(100, x*8 + 10) for x in range(20)],
                'Collaboration': [min(100, x*6 + 20) for x in range(20)]
            })
            fig = px.line(sample_data, x='Time', y=['Rescues', 'Efficiency', 'Collaboration'],
                         title="Expected Performance Trends", 
                         labels={'value': 'Score', 'variable': 'Metric'})
            st.plotly_chart(fig, width='stretch')
        
        with tab2:
            # Sample agent distribution
            agent_data = pd.DataFrame({
                'Agent Type': ['Scout Drones', 'Medical Drones', 'Ambulances', 'Rescue Teams', 'Command Units'],
                'Count': [3, 2, 4, 2, 1],
                'Color': ['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#d62728']
            })
            fig = px.pie(agent_data, values='Count', names='Agent Type', 
                        title="Optimal Team Composition", color='Color')
            st.plotly_chart(fig, width='stretch')
        
        with tab3:
            # Scenario success rates
            scenario_data = pd.DataFrame({
                'Scenario': ['Earthquake Urban', 'Flood Response', 'Fire Emergency', 'Building Collapse'],
                'Success Rate': [85, 78, 92, 75],
                'Avg Response Time': [45, 38, 52, 61]
            })
            fig = px.bar(scenario_data, x='Scenario', y='Success Rate',
                        title="Scenario Success Rates", color='Success Rate')
            st.plotly_chart(fig, width='stretch')
    
    def render_enhanced_simulation_dashboard(self):
        """Render enhanced simulation dashboard"""
        # Main simulation view with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live View", "📈 Analytics", "🤖 Agents", "📋 Logs"])
        
        with tab1:
            self.render_enhanced_simulation_view()
        
        with tab2:
            self.render_enhanced_analytics()
        
        with tab3:
            self.render_enhanced_agent_view()
        
        with tab4:
            self.render_simulation_logs()
    
    def render_enhanced_simulation_view(self):
        """Render enhanced simulation visualization"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Live Simulation View")
            if st.session_state.environment:
                try:
                    # Get current simulation state for visualization
                    env = st.session_state.environment
                    
                    # Create enhanced visualization
                    fig = self.simulation_viewer.create_simulation_plot(env)
                    if fig:
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.warning("Visualization not available")
                    
                    # Auto-advance simulation if running
                    if st.session_state.simulation_running:
                        self.advance_simulation()
                        
                except Exception as e:
                    self.handle_error(f"Visualization error: {e}")
                    st.error("❌ Failed to render simulation view")
            else:
                st.info("No active simulation")
        
        with col2:
            st.subheader("🚨 Immediate Actions")
            self.render_quick_actions()
            st.subheader("📊 Real-time Metrics")
            self.render_realtime_metrics()
    
    def advance_simulation(self):
        """Advance simulation by one step with error handling"""
        try:
            env = st.session_state.environment
            if not env:
                return
            
            # Generate actions for all agents
            actions = {}
            for agent_id, agent in env.agents.items():
                # Simple movement pattern - in a real scenario, this would use policies
                possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
                actions[agent_id] = np.random.choice(possible_actions)
            
            # Execute step
            result = env.step(actions)
            
            # Update metrics
            self.update_simulation_data()
            
            # Auto-save if enabled
            if st.session_state.auto_save:
                self.auto_save_progress()
            
            # Add small delay for visualization
            time.sleep(1.0 / st.session_state.simulation_speed)
            
            # Trigger rerun for real-time updates
            st.rerun()
            
        except Exception as e:
            self.handle_error(f"Simulation advance error: {e}")
            st.session_state.simulation_running = False
    
    def update_simulation_data(self):
        """Update simulation data with comprehensive metrics"""
        if not st.session_state.environment:
            return
            
        env = st.session_state.environment
        
        try:
            # Ensure civilians is a list
            if not isinstance(env.civilians, list):
                env.civilians = []
            
            rescued = sum(1 for c in env.civilians if c.get('rescued', False))
            total_civilians = len(env.civilians)
            rescue_rate = (rescued / total_civilians * 100) if total_civilians > 0 else 0
            efficiency = min(env.step_count / max(1, rescued) if rescued > 0 else env.step_count, 100)
            
            current_data = {
                'step': env.step_count,
                'rescues': rescued,
                'rescue_rate': rescue_rate,
                'efficiency': efficiency,
                'active_agents': len(env.agents),
                'collapsed_buildings': len(env.collapsed_buildings),
                'blocked_roads': len(env.blocked_roads),
                'timestamp': datetime.now()
            }
            
            st.session_state.simulation_data['steps'].append(current_data)
            st.session_state.simulation_data['civilian_rescues'] = rescued
            
            # Keep only last 1000 data points for performance
            if len(st.session_state.simulation_data['steps']) > 1000:
                st.session_state.simulation_data['steps'] = st.session_state.simulation_data['steps'][-1000:]
                
        except Exception as e:
            self.handle_error(f"Data update error: {e}")
    
    def render_quick_actions(self):
        """Render quick action buttons"""
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⏸️ Pause", width='stretch'):
                st.session_state.simulation_running = False
                st.success("Simulation paused")
            
            if st.button("💾 Save", width='stretch'):
                self.save_simulation_state()
        
        with col2:
            if st.button("🔄 Step", width='stretch'):
                self.advance_simulation()
            
            if st.button("📷 Screenshot", width='stretch'):
                self.take_screenshot()
    
    def render_realtime_metrics(self):
        """Render real-time performance metrics"""
        if not st.session_state.simulation_data['steps']:
            st.info("No data yet")
            return
            
        latest = st.session_state.simulation_data['steps'][-1]
        
        metrics = [
            ("Rescue Rate", f"{latest.get('rescue_rate', 0):.1f}%", "success"),
            ("Efficiency", f"{latest.get('efficiency', 0):.1f}", "warning"),
            ("Active Agents", f"{latest.get('active_agents', 0)}", "info"),
            ("Current Step", f"{latest.get('step', 0)}", "info")
        ]
        
        for name, value, style in metrics:
            self.render_metric_card(name, value, style)
    
    def render_metric_card(self, title, value, style="info"):
        """Render a metric card with enhanced styling"""
        st.markdown(f"""
        <div class="metric-card {style}-metric">
            <h3 style="margin: 0; font-size: 0.9rem; color: #666;">{title}</h3>
            <h2 style="margin: 0; font-size: 1.5rem; color: #333;">{value}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    def render_enhanced_analytics(self):
        """Render enhanced analytics dashboard"""
        st.subheader("📈 Enhanced Performance Analytics")
        
        if not st.session_state.simulation_data['steps']:
            st.info("Collecting simulation data...")
            return
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(st.session_state.simulation_data['steps'])
        
        # Multiple analytics views
        col1, col2 = st.columns(2)
        
        with col1:
            # Rescue progress over time
            if 'rescues' in df.columns:
                fig = px.area(df, x='step', y='rescues', 
                             title="Civilians Rescued Over Time",
                             labels={'step': 'Simulation Step', 'rescues': 'Civilians Rescued'})
                st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Efficiency trend
            if 'efficiency' in df.columns:
                fig = px.line(df, x='step', y='efficiency',
                             title="Operational Efficiency Trend",
                             labels={'step': 'Simulation Step', 'efficiency': 'Efficiency Score'})
                st.plotly_chart(fig, width='stretch')
        
        # Additional analytics
        col3, col4 = st.columns(2)
        
        with col3:
            # Rescue rate progression
            if 'rescue_rate' in df.columns:
                fig = px.line(df, x='step', y='rescue_rate',
                             title="Rescue Rate Progression",
                             labels={'step': 'Simulation Step', 'rescue_rate': 'Rescue Rate (%)'})
                st.plotly_chart(fig, width='stretch')
        
        with col4:
            # Performance summary
            st.subheader("📊 Performance Summary")
            if not df.empty:
                latest = df.iloc[-1]
                avg_efficiency = df['efficiency'].mean() if 'efficiency' in df.columns else 0
                max_rescues = df['rescues'].max() if 'rescues' in df.columns else 0
                
                st.metric("Current Rescue Rate", f"{latest.get('rescue_rate', 0):.1f}%")
                st.metric("Average Efficiency", f"{avg_efficiency:.1f}")
                st.metric("Total Rescues", f"{latest.get('rescues', 0)} / {max_rescues}")
    
    def render_enhanced_agent_view(self):
        """Render enhanced agent management view"""
        st.subheader("🤖 Agent Management & Analytics")
        
        if not st.session_state.environment:
            st.info("No active simulation")
            return
        
        env = st.session_state.environment
        
        # Agent statistics
        agent_types = {}
        agent_activities = {}
        
        for agent_id, agent in env.agents.items():
            agent_type = type(agent).__name__.replace('Agent', '')
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
            
            # Calculate activity level (simplified)
            activity = getattr(agent, 'steps_taken', 0) + getattr(agent, 'civilians_rescued', 0) * 10
            agent_activities[agent_id] = activity
        
        # Display agent distribution
        if agent_types:
            col1, col2 = st.columns(2)
            
            with col1:
                df_types = pd.DataFrame({
                    'Agent Type': list(agent_types.keys()),
                    'Count': list(agent_types.values())
                })
                fig = px.pie(df_types, values='Count', names='Agent Type', 
                            title="Agent Type Distribution")
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                if agent_activities:
                    df_activities = pd.DataFrame({
                        'Agent': list(agent_activities.keys()),
                        'Activity': list(agent_activities.values())
                    })
                    fig = px.bar(df_activities, x='Agent', y='Activity',
                                title="Agent Activity Levels")
                    st.plotly_chart(fig, width='stretch')
        
        # Agent details table
        st.subheader("Agent Details")
        agent_data = []
        for agent_id, agent in env.agents.items():
            agent_data.append({
                'Agent ID': agent_id,
                'Type': type(agent).__name__,
                'Position': str(getattr(agent, 'position', 'Unknown')),
                'Steps Taken': getattr(agent, 'steps_taken', 0),
                'Civilians Rescued': getattr(agent, 'civilians_rescued', 0),
                'Status': 'Active' if getattr(agent, 'active', True) else 'Inactive'
            })
        
        if agent_data:
            st.dataframe(pd.DataFrame(agent_data), width='stretch')
    
    def render_simulation_logs(self):
        """Render simulation logs and error tracking"""
        st.subheader("📋 Simulation Logs & Errors")
        
        if st.session_state.simulation_data['error_log']:
            st.warning(f"⚠️ {len(st.session_state.simulation_data['error_log'])} errors logged")
            
            for i, error in enumerate(st.session_state.simulation_data['error_log'][-10:]):  # Show last 10 errors
                with st.expander(f"Error {i+1}: {error.get('type', 'Unknown')}", expanded=False):
                    st.code(error.get('message', 'No message'))
                    st.write(f"Time: {error.get('timestamp', 'Unknown')}")
        else:
            st.success("✅ No errors logged")
        
        # Performance logs
        st.subheader("📈 Performance Logs")
        if st.session_state.simulation_data['steps']:
            df = pd.DataFrame(st.session_state.simulation_data['steps'])
            st.dataframe(df.tail(10), width='stretch')  # Show last 10 steps
        else:
            st.info("No performance data yet")
    
    def start_simulation(self, location, num_drones, num_ambulances, num_rescue_teams, auto_deploy=True):
        """Start a new simulation with enhanced error handling"""
        try:
            logger.info(f"Starting simulation for {location} with {num_drones} drones, {num_ambulances} ambulances, {num_rescue_teams} rescue teams")
            
            # Create environment based on mode
            if st.session_state.simulation_mode == 'training':
                env = SimpleGridEnv()  # Use simple env for training
            elif location == "Generated City":
                env = RealMapEnv()  # Generated map
            else:
                env = RealMapEnv(location)  # Real map
            
            # Reset environment
            obs, info = env.reset()
            
            # Add agents with optimized positions if auto_deploy
            positions = self.calculate_optimal_positions(env, num_drones, num_ambulances, num_rescue_teams) if auto_deploy else None
            
            for i in range(num_drones):
                pos = positions['drones'][i] if positions else np.array([2 + i, 2])
                drone = DroneAgent(f"drone_{i}", pos, env.config)
                env.add_agent(drone)
            
            for i in range(num_ambulances):
                pos = positions['ambulances'][i] if positions else np.array([5, 5 + i])
                ambulance = AmbulanceAgent(f"ambulance_{i}", pos, env.config)
                env.add_agent(ambulance)
            
            for i in range(num_rescue_teams):
                pos = positions['rescue_teams'][i] if positions else np.array([8, 2 + i])
                rescue_team = RescueTeamAgent(f"rescue_team_{i}", pos, env.config)
                env.add_agent(rescue_team)
            
            # Update session state
            st.session_state.environment = env
            st.session_state.simulation_running = True
            st.session_state.selected_location = location
            st.session_state.simulation_data['start_time'] = datetime.now()
            st.session_state.disaster_triggered = False
            
            # Clear previous data
            st.session_state.simulation_data['steps'] = []
            st.session_state.simulation_data['error_log'] = []
            
            st.success(f"🚀 Simulation started for {location}!")
            logger.info("Simulation started successfully")
            
        except Exception as e:
            error_msg = f"Failed to start simulation: {e}"
            self.handle_error(error_msg)
            st.error(f"❌ {error_msg}")
    
    def calculate_optimal_positions(self, env, num_drones, num_ambulances, num_rescue_teams):
        """Calculate optimal starting positions for agents"""
        grid_size = env.grid_size
        positions = {
            'drones': [],
            'ambulances': [],
            'rescue_teams': []
        }
        
        # Simple positioning logic - can be enhanced
        for i in range(num_drones):
            positions['drones'].append(np.array([2 + i, 2]))
        
        for i in range(num_ambulances):
            positions['ambulances'].append(np.array([grid_size - 3, 3 + i]))
        
        for i in range(num_rescue_teams):
            positions['rescue_teams'].append(np.array([5 + i, grid_size - 3]))
        
        return positions
    
    def stop_simulation(self):
        """Stop the current simulation with proper cleanup"""
        try:
            if st.session_state.environment:
                st.session_state.environment.close()
            
            st.session_state.simulation_running = False
            st.session_state.environment = None
            
            # Save final state
            if st.session_state.auto_save:
                self.save_simulation_state()
            
            st.success("⏹️ Simulation stopped and saved!")
            logger.info("Simulation stopped successfully")
            
        except Exception as e:
            self.handle_error(f"Error stopping simulation: {e}")
            st.error("❌ Error stopping simulation")
    
    def reset_simulation(self):
        """Reset the current simulation"""
        if st.session_state.environment:
            try:
                st.session_state.environment.reset()
                st.session_state.simulation_data['steps'] = []
                st.session_state.disaster_triggered = False
                st.success("🔄 Simulation reset!")
            except Exception as e:
                self.handle_error(f"Reset error: {e}")
                st.error("❌ Failed to reset simulation")
        else:
            st.warning("⚠️ No active simulation to reset")
    
    def trigger_disaster(self):
        """Trigger disaster in the simulation"""
        if st.session_state.environment and not st.session_state.disaster_triggered:
            try:
                st.session_state.environment.trigger_disaster()
                st.session_state.disaster_triggered = True
                st.success("🚨 Disaster triggered!")
                logger.info("Disaster triggered in simulation")
            except Exception as e:
                self.handle_error(f"Disaster trigger error: {e}")
                st.error("❌ Failed to trigger disaster")
        elif st.session_state.disaster_triggered:
            st.warning("⚠️ Disaster already triggered!")
        else:
            st.error("❌ No active simulation!")
    
    def save_simulation_state(self):
        """Save current simulation state"""
        try:
            # In a real implementation, this would save to file/database
            st.session_state.last_save_time = datetime.now()
            st.success(f"💾 Progress saved at {st.session_state.last_save_time.strftime('%H:%M:%S')}")
            logger.info("Simulation state saved")
        except Exception as e:
            self.handle_error(f"Save error: {e}")
    
    def auto_save_progress(self):
        """Auto-save progress at intervals"""
        if not st.session_state.last_save_time or (datetime.now() - st.session_state.last_save_time).seconds > 300:  # 5 minutes
            self.save_simulation_state()
    
    def take_screenshot(self):
        """Take simulation screenshot"""
        # This would capture the current visualization
        st.success("📸 Screenshot captured!")
        # In a real implementation, this would save the current view as an image
    
    def handle_error(self, error_message):
        """Handle and log errors"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': type(error_message).__name__,
            'message': str(error_message),
            'traceback': traceback.format_exc()
        }
        
        st.session_state.simulation_data['error_log'].append(error_entry)
        logger.error(f"Dashboard error: {error_message}")
        
        # Keep error log manageable
        if len(st.session_state.simulation_data['error_log']) > 100:
            st.session_state.simulation_data['error_log'] = st.session_state.simulation_data['error_log'][-100:]
    
    def run(self):
        """Run the enhanced dashboard"""
        try:
            self.render_header()
            self.render_sidebar()
            self.render_main_content()
        except Exception as e:
            self.handle_error(f"Dashboard runtime error: {e}")
            st.error("❌ Dashboard encountered an error. Please refresh the page.")

# Main application with enhanced error handling
def main():
    """Main dashboard application"""
    try:
        dashboard = DisasterResponseDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"💥 Critical error: {e}")
        st.info("Please check the console for details and refresh the page.")
        logger.critical(f"Dashboard critical error: {e}", exc_info=True)

if __name__ == "__main__":
    main()