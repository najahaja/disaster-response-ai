#!/usr/bin/env python3
"""
Enhanced Main Dashboard Application for Disaster Response AI - Week 6
Streamlit-based dashboard with login portal for admin and viewer roles
"""
from stable_baselines3 import PPO
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
import hashlib

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

class LoginSystem:
    """Simple login system for admin and viewer roles"""
    
    def __init__(self):
        # Predefined credentials (in production, use secure storage)
        self.users = {
            "admin": {
                "password": self._hash_password("Admin@123"),
                "role": "admin"
            },
            "viewer": {
                "password": self._hash_password("Viewer@123"),
                "role": "viewer"
            }
        }
    
    def _hash_password(self, password):
        """Simple password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username, password):
        """Authenticate user"""
        if username in self.users:
            hashed_password = self._hash_password(password)
            if self.users[username]["password"] == hashed_password:
                return True, self.users[username]["role"]
        return False, None
    
    def is_admin(self, role):
        """Check if role is admin"""
        return role == "admin"
    
    def is_viewer(self, role):
        """Check if role is viewer"""
        return role == "viewer"

class DisasterResponseDashboard:
    """
    Enhanced dashboard class for Disaster Response AI simulation - Week 6
    """
    
    def __init__(self):
        self.login_system = LoginSystem()
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
        /* Global Styles */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            font-family: 'Inter', sans-serif;
        }
        
        /* Header Styling */
        .main-header {
            font-size: 3.5rem;
            background: linear-gradient(120deg, #ff6b6b, #556270);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-weight: 800;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            letter-spacing: -1px;
        }
        
        /* Login Form Styling */
        .login-form { 
        max-width: 500px; 
        margin: 20px auto; 
        padding: 1rem 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        text-align: center; }
        

        
        .role-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }
        
        .admin-badge {
            background-color: #ff6b6b;
            color: white;
        }
        
        .viewer-badge {
            background-color: #0984e3;
            color: white;
        }
        
        /* Card Styling */
        .metric-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            margin-bottom: 1rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
        }
        
        /* Metric Borders */
        .success-metric { border-left: 5px solid #00b894; }
        .warning-metric { border-left: 5px solid #fdcb6e; }
        .danger-metric { border-left: 5px solid #ff7675; }
        .info-metric { border-left: 5px solid #0984e3; }
        
        /* Button Styling */
        .stButton button {
            width: 100%;
            border-radius: 12px;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 14px rgba(0,0,0,0.15);
        }
        
        /* Disabled Button */
        .stButton button:disabled {
            background-color: #cccccc !important;
            cursor: not-allowed;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0;
        }
        
        /* Headings */
        h1, h2, h3 {
            color: #2d3436;
            font-weight: 700;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize session state variables with enhanced tracking"""
        default_state = {
            'authenticated': False,
            'username': None,
            'user_role': None,
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
    
    def render_login(self):
        """Render login form - Corrected Version"""
        st.markdown('<h1 class="main-header">🚨 Disaster Response AI Dashboard</h1>', 
                    unsafe_allow_html=True)

        # Create 3 columns: Empty | Form | Empty
        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            st.markdown("""
            <div class="login-form">
                <h3>🔐 Login Portal</h3>
                <p>Please enter your credentials to access the dashboard</p>
            </div>
            """, unsafe_allow_html=True)

            with st.form("login_form"):
                # Inputs for username and password
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                
                # Submit button
                submit_button = st.form_submit_button("Login", use_container_width=True)
                
            # Handle the actions when the login button is pressed
            if submit_button:
                if username and password:
                    authenticated, role = self.login_system.authenticate(username, password)
                    if authenticated:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_role = role
                        st.success(f"✅ Welcome {username}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter both username and password")

            # Removed the 'if demo_button:' block completely to prevent the error

            # Expandable Note Section
            with st.expander("Note"):
                st.markdown("""
                Admin can trigger disasters and modify settings. Viewer can only view.
                """)

        
        
    
    def render_header(self):
        """Render a two-row header with fixed button styling"""
        
        st.markdown("""
            <style>
            /* --- HEADER TEXT --- */
            .main-header {
                font-size: 46px !important;
                font-weight: 700 !important;
                margin: 0 !important;
            }
            .user-info-box {
                text-align: right;
                line-height: 1.2;
                margin-bottom: 5px;
            }
            .role-badge {
                padding: 2px 6px;
                border-radius: 10px;
                font-size: 0.75rem;
                color: white !important;
                font-weight: bold;
                text-transform: uppercase;
                margin-left: 5px;
            }
            .admin-badge { background-color: #ff4b4b; }
            .viewer-badge { background-color: #007bff; }

            /* --- LOGOUT BUTTON (FIXED ALIGNMENT) --- */
            .logout-container {
                display: flex !important;
                justify-content: flex-end !important;
                width: 100% !important;
                padding-top: 10px !important;
            }
            .logout-container div[data-testid="stButton"] > button {
                width: 120px !important;
                height: 40px !important;
                background-color: white !important;
                color: #31333F !important;
                border: 1px solid #dcdcdc !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
                transition: 0.3s ease !important;
            }
            .logout-container div[data-testid="stButton"] > button:hover {
                background-color: #ff4b4b !important;
                color: white !important;
                border-color: #ff4b4b !important;
                /* Removed margin-right to prevent 'jumping' */
            }

            /* --- METRICS SCALING --- */
            [data-testid="stMetricLabel"] p {
                font-size: 1.4rem !important; /* Slightly reduced for better fit */
                font-weight: 600 !important;
                padding-left: 10px !important;
            }
            [data-testid="stMetricValue"] > div {
                font-size: 2.0rem !important;
                padding-left: 10px !important;
                color: #1f77b4 !important;
            }
            /* Specialized padding for the Run Time column */
            [data-testid="column"]:nth-child(4) [data-testid="stMetric"] {
                padding-right: 20px !important;
                text-align: right !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # --- ROW 1: Title and Login Info ---
        row1_col1, row1_col2 = st.columns([7, 1]) # Adjusted ratio for better spacing
        
        with row1_col1:
            st.markdown('<h1 class="main-header">🚨 Disaster Response AI Dashboard</h1>', unsafe_allow_html=True)
        
        with row1_col2:
            role = st.session_state.user_role
            badge_class = "admin-badge" if role == "admin" else "viewer-badge"
            st.markdown(f"""
                <div class="user-info-box">
                    <small style="color: gray;">Logged in as</small><br>
                    <strong>{st.session_state.username}</strong>
                    <span class="role-badge {badge_class}">{role}</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="logout-container">', unsafe_allow_html=True)
            if st.button("LOGOUT", key="top_logout"):
                st.session_state.authenticated = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # --- ROW 2: The 4 Metrics ---
        m_col1, m_col2, m_col3, m_col4 = st.columns([2,2,1,2])

        with m_col1:
            status_icon = "🟢" if st.session_state.simulation_running else "🔴"
            status_text = "Running" if st.session_state.simulation_running else "Stopped"
            st.metric("Status", f"{status_icon} {status_text}")

        with m_col2:
            if not st.session_state.simulation_running:
                display_env = "Not Loaded"
            else:
                raw_loc = st.session_state.get('selected_location', '')
                display_env = "Generated" if raw_loc == "Generated City" else "Real Map"
            st.metric("Environment", display_env)

        with m_col3:
            rescued = sum(1 for c in st.session_state.environment.civilians if c.get('rescued')) if st.session_state.environment else 0
            st.metric("Rescued", f"{rescued} 🎯")

        with m_col4:
            if st.session_state.simulation_running and st.session_state.simulation_data['start_time']:
                from datetime import datetime
                elapsed = (datetime.now() - st.session_state.simulation_data['start_time']).total_seconds()
                time_display = f"{int(elapsed)}s"
            else:
                time_display = "0s"
            st.metric("Run Time", f"{time_display} ⏱️")

        st.divider()
        
    def render_sidebar(self):
        """Render enhanced sidebar with role-based permissions"""
        with st.sidebar:
            st.header("🎮 Simulation Controls")
            
          # --- User Info Display ---
            # Using a single markdown block for better alignment
            role_color = "#FF4B4B" if st.session_state.user_role == "admin" else "#1F77B4"
            
            st.markdown(f"""
                <div style="background-color: rgba(151, 166, 195, 0.1); padding: 10px; border-radius: 5px;">
                    <span style="font-weight: bold;">👤 User:</span> {st.session_state.username}<br>
                    <span style="font-weight: bold;">🛡️ Role:</span> <span style="color: {role_color}; font-weight: bold;">{st.session_state.user_role.capitalize()}</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Simulation Mode Selection
            st.subheader("🔧 Simulation Mode")
            mode = st.selectbox(
                "Select Mode",
                options=["Basic", "Training"],
                index=0,
                help="Basic: Simple simulation\nTraining: ML training mode",
                disabled=st.session_state.user_role == "viewer"  # Disable for viewer
            )
            st.session_state.simulation_mode = mode.lower()
            
            # Environment selection with enhanced options
            st.subheader("🗺️ Environment Setup")
            location_options = [
                "Lahore",
                "Karachi", 
                "Islamabad",
                "Tokyo",
                "New York", 
                "Generated City"
            ]
            selected_location = st.selectbox(
                "Select Location",
                options=location_options,
                index=0,
                disabled=st.session_state.user_role == "viewer"  # Disable for viewer
            )
            
            # Enhanced Agent configuration
            st.subheader("🤖 Agent Configuration")
            num_drones = st.slider("Drones 🛸", 1, 8, 2, 
                                    help="Aerial reconnaissance units",
                                    disabled=st.session_state.user_role == "viewer")  # Disable for viewer
            num_ambulances = st.slider("Ambulances 🚑", 1, 6, 2,
                                        help="Medical transport units",
                                        disabled=st.session_state.user_role == "viewer")  # Disable for viewer
            
        
            num_rescue_teams = st.slider("Rescue Teams 👷", 1, 4, 1,
                                        help="Ground rescue units",
                                        disabled=st.session_state.user_role == "viewer")  # Disable for viewer
            # ADD THIS LINE - Dynamic civilian count
            num_civilians = st.slider("Civilians 👥", 1, 20, 8,
                                        help="Number of civilians to rescue",
                                        disabled=st.session_state.user_role == "viewer")  # Disable for viewer
            
            auto_deploy = st.checkbox("Auto-deploy", value=True,
                                    help="Automatically deploy agents at optimal positions",
                                    disabled=st.session_state.user_role == "viewer")  # Disable for viewer
            st.session_state.num_civilians = num_civilians
            # Enhanced Simulation controls
            st.subheader("🎯 Simulation Control")
            
            # control_col1, control_col2 = st.columns(2)
            # with control_col1:
            start_disabled = st.session_state.user_role == "viewer"
            if st.button("🚀 Start Simulation", use_container_width=True, 
                           help="Start new simulation with current configuration",
                           disabled=start_disabled):
                    self.start_simulation(selected_location, num_drones, num_ambulances, num_rescue_teams, num_civilians, auto_deploy)
            
            # with control_col2:
            stop_disabled = st.session_state.user_role == "viewer" or not st.session_state.simulation_running
            if st.button("⏹️ Stop Simulation", use_container_width=True,
                           help="Stop current simulation",
                           disabled=stop_disabled):
                    self.stop_simulation()
            
            # Additional control buttons - only for admin
            if st.session_state.user_role == "admin":
                if st.button("🚨 Trigger Disaster", use_container_width=True,
                           help="Trigger disaster scenario in current simulation"):
                    self.trigger_disaster()
                
                if st.button("🔄 Reset Simulation", use_container_width=True,
                           help="Reset simulation to initial state"):
                    self.reset_simulation()
            else:
                # For viewer, show disabled buttons or info
                st.button("🚨 Trigger Disaster", use_container_width=True, disabled=True,
                         help="Admin permission required")
                st.button("🔄 Reset Simulation", use_container_width=True, disabled=True,
                         help="Admin permission required")
            
            # Enhanced settings
            st.subheader("⚙️ Simulation Settings")
            sim_speed = st.slider("Simulation Speed", 1, 20, 5,
                                help="Faster speed may reduce visualization quality")
            st.session_state.simulation_speed = sim_speed
            
            auto_save = st.checkbox("Auto-save progress", value=True,
                                  help="Automatically save simulation state",
                                  disabled=st.session_state.user_role == "viewer")  # Disable for viewer
            st.session_state.auto_save = auto_save
            
            # Real-time metrics in sidebar
            st.subheader("📊 Live Metrics")
            if st.session_state.simulation_running and st.session_state.environment:
                self.render_live_metrics()
            
            # Session info
            # Session info
            st.subheader("💾 Session Info")
            
            # Use a container to "lock" the position
            info_container = st.container()
            with info_container:
                st.write(f"Session ID: `{st.session_state.session_id}`")
                if st.session_state.simulation_data['start_time']:
                    st.write(f"Simulation Start Time: {st.session_state.simulation_data['start_time'].strftime('%H:%M:%S')}")
                
                # ADD THIS: Show disaster time here too, so it's all in one place
                if st.session_state.get('disaster_triggered'):
                    d_time = st.session_state.simulation_data.get('disaster_time')
                    if d_time:
                        st.error(f"🔥 Disaster Triggered: {d_time.strftime('%H:%M:%S')}")
    def render_live_metrics(self):
        """Render enhanced live metrics in sidebar"""
        env = st.session_state.environment
        if not env:
            return
        
        # Check if disaster has been triggered
        if st.session_state.disaster_triggered:
            # After disaster: show actual civilian metrics
            rescued = sum(1 for c in env.civilians if c.get('rescued', False))
            total_civilians = len(env.civilians)
            target_civilians = st.session_state.get('num_civilians', 8)
            
            # Use whichever is smaller (in case some civilians couldn't be placed)
            effective_total = min(total_civilians, target_civilians)
            rescue_rate = (rescued / effective_total * 100) if effective_total > 0 else 0
            
            metrics = {
                "Step": f"{env.step_count} 📈",
                "Status": "🚨 DISASTER",
                "Active Agents": f"{len(env.agents)} 🤖",
                "Civilians": f"{total_civilians} 👥",
                "Rescued": f"{rescued} ✅ ({rescue_rate:.1f}%)",
                "Remaining": f"{effective_total - rescued} ⚠️",
            }
            
            # Progress bar for rescue
            st.progress(rescued / effective_total if effective_total > 0 else 0, 
                    text=f"Rescue: {rescued}/{effective_total}")
        else:
            # Before disaster: show readiness metrics
            metrics = {
                "Step": f"{env.step_count} 📈",
                "Status": "✅ READY",
                "Active Agents": f"{len(env.agents)} 🤖",
                "Civilians": "0 👥",
                "Disaster": "⏳ Waiting",
                "Ready Civilians": f"{st.session_state.get('num_civilians', 8)} ⚡",
            }
            
            # Show info about pending disaster
            st.info("⚠️ Trigger disaster to spawn civilians")
        
        for metric, value in metrics.items():
            st.metric(metric, value)
    
    def render_main_content(self):
        """Render main dashboard content with enhanced layout"""
        if st.session_state.environment is None:
            self.render_enhanced_welcome_screen()
        else:
            self.render_enhanced_simulation_dashboard()
    
    def render_enhanced_welcome_screen(self):
        """Render enhanced welcome screen"""
        
            
        
        st.markdown("""
        ## 🌟 Welcome to Disaster Response AI Dashboard
        """)
        st.divider()    
        st.success("""
        ### 🧭 Simulation Legend:
        
        **🛰️ Active Agents & Targets:**
        * 🟢 **Rescue Teams**: Specialized ground coordination units.
        * 🔵 **Drones**: Multi-agent aerial surveillance and mapping.
        * 🔴 **Ambulances**: Medical transport for rescued civilians.
        * 🟡 **Civilians**: Targets requiring immediate assistance.

        **🏘️ Environment & Hazards:**
        * 🟤 **Buildings**: Standard urban structures and housing.
        * 🔘 **Roads**: Navigable street and transport network.
        * ⚪ **Hospitals**: Safe zones and medical delivery points.
        * 🟠 **Collapsed Buildings**: Significant structural failures or damage.
        * 🛑 **Road Blocks**: Blocked routes or debris-filled paths.
        """)
        st.divider()    
        st.markdown(""" 
        ### Role Information:
        """)
        
        # Display role-specific information
        if st.session_state.user_role == "admin":
            st.success("""
            **👑 Admin Privileges:**
            - 🚀 Start/Stop simulations
            - 🚨 Trigger disasters
            - ⚙️ Modify all settings
            - 🤖 Configure agents
            - 📊 Full control over all features
            """)
        else:
            st.info("""
            **👁️ Viewer Privileges:**
            - 👀 View simulations
            - 📊 Monitor metrics
            - 📈 Analyze performance
            - ℹ️ Read-only access to all data
            """)
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            if st.session_state.user_role == "admin":
                st.markdown("""
                1. **Select Simulation Mode** in sidebar
                2. **Choose Location** (real map or generated)
                3. **Configure Your Team** (drones, ambulances, rescue teams)
                4. **Click 'Start Simulation'**
                5. **Trigger Disaster** when ready
                6. **Monitor Performance** in real-time
                """)
            else:
                st.markdown("""
                1. **Wait for Admin** to start simulation
                2. **Observe Simulation** in real-time
                3. **Monitor Metrics** in sidebar
                4. **Analyze Performance** in analytics tab
                5. **View Agent Activities** in agents tab
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
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Sample agent distribution
            agent_data = pd.DataFrame({
                'Agent Type': ['Scout Drones', 'Medical Drones', 'Ambulances', 'Rescue Teams', 'Command Units'],
                'Count': [3, 2, 4, 2, 1],
                'Color': ['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#d62728']
            })
            fig = px.pie(agent_data, values='Count', names='Agent Type', 
                        title="Optimal Team Composition", color='Color')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Scenario success rates
            scenario_data = pd.DataFrame({
                'Scenario': ['Earthquake Urban', 'Flood Response', 'Fire Emergency', 'Building Collapse'],
                'Success Rate': [85, 78, 92, 75],
                'Avg Response Time': [45, 38, 52, 61]
            })
            fig = px.bar(scenario_data, x='Scenario', y='Success Rate',
                        title="Scenario Success Rates", color='Success Rate')
            st.plotly_chart(fig, use_container_width=True)
    
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
        """Render the live simulation (full‑width map + controls below)."""
        # Display role-based message
        if st.session_state.user_role == "viewer":
            st.info("👁️ **Viewer Mode**: You can observe the simulation in real-time. Admin controls are disabled.")
        
        # ---------- MAP ----------
        st.subheader("🎯 Live Simulation View")
        image_placeholder = st.empty()

        if st.session_state.environment:
            try:
                env = st.session_state.environment
                pygame_surface = env.render()
                if pygame_surface is not None:
                    self.simulation_viewer.update_frame(pygame_surface)

                # Show the latest frame
                self.simulation_viewer.render(image_placeholder)

                # Advance the simulation *and* trigger a rerun so the next frame appears
                if st.session_state.simulation_running:
                    self.advance_simulation()
                    st.rerun()          # <-- crucial line
            except Exception as e:
                self.handle_error(f"Visualization error: {e}")
                st.error("❌ Failed to render simulation view")
        else:
            st.info("No active simulation. Start one from the sidebar.")
            self.simulation_viewer.render(image_placeholder)

        # ---------- CONTROLS & METRICS ----------
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🚨 Immediate Actions")
            self.render_quick_actions()

        with col2:
            st.subheader("📊 Real‑time Metrics")
            self.render_realtime_metrics()
    
    def advance_simulation(self):
        """Advance simulation by one step with error handling"""
        try:
            env = st.session_state.environment
            if not env:
                return

            # --- AI Model Loading ---
            # We load this once per session to save time
            if 'ai_model' not in st.session_state:
                try:
                    from stable_baselines3 import PPO 
                    st.session_state['ai_model'] = PPO.load("models/disaster_response_model")
                except Exception as e:
                    st.warning(f"Could not load AI model: {e}. Using Random Agent.")
                    st.session_state['ai_model'] = None

            model = st.session_state['ai_model']
            use_ai = (model is not None)

            # --- Action Selection ---
            actions = {}
            # Get observation ONCE for all agents
            obs = env._get_gym_observation()
            
            for agent_id, agent in env.agents.items():
                if use_ai:
                    try:
                        # Predict action
                        action, _ = model.predict(obs, deterministic=False)
                        actions[agent_id] = int(action)
                    except Exception as e:
                        # If prediction fails (e.g. size mismatch), fallback to random and warn
                        print(f"AI Prediction Error: {e}")
                        actions[agent_id] = env.action_space.sample()
                else:
                    actions[agent_id] = env.action_space.sample()
            
            # --- Step the Environment ---
            obs, reward, terminated, truncated, info = env.step(actions)
            
            # --- Update Data ---
            self.update_simulation_data()
            
            # --- Check Termination ---
            if terminated or truncated:
                st.session_state.simulation_running = False
                st.balloons()
                st.success("Simulation Complete!")
                time.sleep(2)
                st.rerun()
                return

            # --- Auto-save & Delay ---
            if st.session_state.auto_save:
                self.auto_save_progress()
            
            time.sleep(1.0 / st.session_state.simulation_speed)
            
            # --- CRITICAL: FORCE RERUN ---
            st.rerun()
            
        except Exception as e:
            print("CRITICAL SIMULATION ERROR:")
            print(traceback.format_exc())
            # This catches any other crashes and shows them
            st.error(f"💥 Simulation Crashed: {e}")
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
        """Render quick action buttons with role-based permissions"""
        col1, col2 = st.columns(2)
        
        with col1:
            pause_disabled = st.session_state.user_role == "viewer" or not st.session_state.simulation_running
            if st.button("⏸️ Pause", use_container_width=True, disabled=pause_disabled):
                st.session_state.simulation_running = False
                st.success("Simulation paused")
            
            save_disabled = st.session_state.user_role == "viewer"
            if st.button("💾 Save", use_container_width=True, disabled=save_disabled):
                self.save_simulation_state()
        
        with col2:
            step_disabled = st.session_state.user_role == "viewer" or not st.session_state.environment
            if st.button("🔄 Step", use_container_width=True, disabled=step_disabled):
                self.advance_simulation()
            
            if st.button("📷 Screenshot", use_container_width=True):
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
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Efficiency trend
            if 'efficiency' in df.columns:
                fig = px.line(df, x='step', y='efficiency',
                             title="Operational Efficiency Trend",
                             labels={'step': 'Simulation Step', 'efficiency': 'Efficiency Score'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional analytics
        col3, col4 = st.columns(2)
        
        with col3:
            # Rescue rate progression
            if 'rescue_rate' in df.columns:
                fig = px.line(df, x='step', y='rescue_rate',
                             title="Rescue Rate Progression",
                             labels={'step': 'Simulation Step', 'rescue_rate': 'Rescue Rate (%)'})
                st.plotly_chart(fig, use_container_width=True)
        
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
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if agent_activities:
                    df_activities = pd.DataFrame({
                        'Agent': list(agent_activities.keys()),
                        'Activity': list(agent_activities.values())
                    })
                    fig = px.bar(df_activities, x='Agent', y='Activity',
                                title="Agent Activity Levels")
                    st.plotly_chart(fig, use_container_width=True)
        
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
            st.dataframe(pd.DataFrame(agent_data), use_container_width=True)
    
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
            st.dataframe(df.tail(10), use_container_width=True)  # Show last 10 steps
        else:
            st.info("No performance data yet")
    
    @st.cache_resource
    def load_ai_model(self):
        """Load the trained AI model (Cached for performance)"""
        model_path = "models/disaster_response_model" # It automatically adds .zip
        
        if os.path.exists(model_path + ".zip"):
            try:
                model = PPO.load(model_path)
                logger.info("✅ AI Model loaded successfully!")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return None
        else:
            logger.warning("⚠️ Model file not found. Using random actions.")
            return None
    
    def start_simulation(self, location, num_drones, num_ambulances, num_rescue_teams, num_civilians, auto_deploy=True):
        """Start a new simulation with enhanced error handling"""
        try:
            # Check if user has permission
            if st.session_state.user_role != "admin":
                st.error("❌ Admin permission required to start simulation")
                return
                
            logger.info(f"Starting simulation for {location} with {num_drones} drones, {num_ambulances} ambulances, {num_rescue_teams} rescue teams")
            
            # Store civilian count in session state (for later when disaster is triggered)
            st.session_state.num_civilians = num_civilians
            
            # Create environment based on mode
            if st.session_state.simulation_mode == 'training':
                env = SimpleGridEnv()  # Use simple env for training
            elif location == "Generated City":
                env = RealMapEnv()  # Generated map
            else:
                env = RealMapEnv(location)  # Real map
            
            # Reset environment
            obs, info = env.reset()
            
            # IMPORTANT: Initialize empty civilians list
            if hasattr(env, 'civilians'):
                env.civilians = []  # Start with NO civilians
                logger.info("Initialized empty civilians list (waiting for disaster)")
            
            # Calculate positions only if auto_deploy is True
            positions = None
            if auto_deploy:
                positions = self.calculate_optimal_positions(env, num_drones, num_ambulances, num_rescue_teams)
                if positions is None:
                    logger.warning("⚠️ Using random positions for agents")
            
            # Add drones
            for i in range(num_drones):
                try:
                    if positions and 'drones' in positions and i < len(positions['drones']):
                        pos = positions['drones'][i]
                    else:
                        pos = np.array([2 + i * 2, 2])
                    drone = DroneAgent(f"drone_{i}", pos, env.config)
                    env.add_agent(drone)
                except Exception as e:
                    logger.error(f"Failed to add drone_{i}: {e}")
            
            # Add ambulances
            for i in range(num_ambulances):
                try:
                    if positions and 'ambulances' in positions and i < len(positions['ambulances']):
                        pos = positions['ambulances'][i]
                    else:
                        pos = np.array([5, 5 + i * 2])
                    ambulance = AmbulanceAgent(f"ambulance_{i}", pos, env.config)
                    env.add_agent(ambulance)
                except Exception as e:
                    logger.error(f"Failed to add ambulance_{i}: {e}")
            
            # Add rescue teams
            for i in range(num_rescue_teams):
                try:
                    if positions and 'rescue_teams' in positions and i < len(positions['rescue_teams']):
                        pos = positions['rescue_teams'][i]
                    else:
                        pos = np.array([8, 2 + i * 3])
                    rescue_team = RescueTeamAgent(f"rescue_team_{i}", pos, env.config)
                    env.add_agent(rescue_team)
                except Exception as e:
                    logger.error(f"Failed to add rescue_team_{i}: {e}")
            
            # Update session state
            st.session_state.environment = env
            st.session_state.simulation_running = True
            st.session_state.selected_location = location
            st.session_state.simulation_data['start_time'] = datetime.now()
            st.session_state.disaster_triggered = False  # Important: disaster not triggered yet
            
            # Clear previous data
            st.session_state.simulation_data['steps'] = []
            st.session_state.simulation_data['error_log'] = []
            
            # Show success message (NO civilians yet)
            # st.success(f"🚀 Simulation started for {location}!")
            # st.info(f"• Agents deployed: {num_drones} drones, {num_ambulances} ambulances, {num_rescue_teams} rescue teams")
            # st.info(f"• Civilians: 0 (Will appear when disaster is triggered)")
            # logger.info(f"Simulation started successfully with {len(env.agents)} agents. Waiting for disaster trigger.")
            
        except Exception as e:
            error_msg = f"Failed to start simulation: {e}"
            self.handle_error(error_msg)
            st.error(f"❌ {error_msg}")
            logger.error(traceback.format_exc())
    
    def calculate_optimal_positions(self, env, num_drones, num_ambulances, num_rescue_teams):
        """Find valid ROAD/OPEN positions for agents so they don't get stuck in walls"""
        try:
            # Check if environment has required attributes
            if not hasattr(env, 'grid_size'):
                logger.error("Environment doesn't have grid_size attribute")
                return None
                
            grid_size = env.grid_size
            valid_spots = []
            
            # Check if environment has grid
            if hasattr(env, 'grid') and hasattr(env, 'cell_types'):
                cell_types = env.cell_types
                # Find all cells that are ROAD, OPEN_SPACE, or HOSPITAL (valid for ground vehicles)
                for y in range(grid_size):
                    for x in range(grid_size):
                        if y < grid_size and x < grid_size:  # Safety check
                            cell_value = env.grid[y, x]
                            # Use actual cell_types from config
                            if cell_value in [
                                cell_types.get('ROAD', 1),
                                cell_types.get('OPEN_SPACE', 3),
                                cell_types.get('HOSPITAL', 4)
                            ]:
                                valid_spots.append(np.array([x, y]))
            else:
                # If no grid, create some default positions
                logger.warning("Environment doesn't have grid attribute, using default positions")
                for y in range(2, grid_size-2, 3):
                    for x in range(2, grid_size-2, 3):
                        valid_spots.append(np.array([x, y]))
            
            # If no valid spots found, create fallback positions
            if not valid_spots:
                logger.warning("No valid spots found, using fallback positions")
                # Create positions around the edges
                for i in range(max(num_drones, num_ambulances, num_rescue_teams) * 2):
                    x = np.random.randint(2, grid_size-2)
                    y = np.random.randint(2, grid_size-2)
                    valid_spots.append(np.array([x, y]))
            
            import random
            random.shuffle(valid_spots)
            
            positions = {
                'drones': [],
                'ambulances': [],
                'rescue_teams': []
            }
            
            # Assign spots for drones
            for i in range(num_drones):
                if i < len(valid_spots):
                    positions['drones'].append(valid_spots[i])
                else:
                    # Fallback position
                    positions['drones'].append(np.array([2 + i * 2, 2]))
            
            # Assign spots for ambulances (skip spots used by drones)
            start_idx = num_drones
            for i in range(num_ambulances):
                idx = start_idx + i
                if idx < len(valid_spots):
                    positions['ambulances'].append(valid_spots[idx])
                else:
                    # Fallback position
                    positions['ambulances'].append(np.array([5, 5 + i * 2]))
            
            # Assign spots for rescue teams (skip spots used by drones and ambulances)
            start_idx = num_drones + num_ambulances
            for i in range(num_rescue_teams):
                idx = start_idx + i
                if idx < len(valid_spots):
                    positions['rescue_teams'].append(valid_spots[idx])
                else:
                    # Fallback position
                    positions['rescue_teams'].append(np.array([8, 2 + i * 3]))
            
            logger.info(f"Calculated positions: {len(positions['drones'])} drones, {len(positions['ambulances'])} ambulances, {len(positions['rescue_teams'])} rescue teams")
            return positions
            
        except Exception as e:
            logger.error(f"Error in calculate_optimal_positions: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def stop_simulation(self):
        """Stop the current simulation with proper cleanup"""
        try:
            # Check if user has permission
            if st.session_state.user_role != "admin":
                st.error("❌ Admin permission required to stop simulation")
                return
                
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
        # Check if user has permission
        if st.session_state.user_role != "admin":
            st.error("❌ Admin permission required to reset simulation")
            return
            
        if st.session_state.environment:
            try:
                env = st.session_state.environment
                
                # Reset the environment
                env.reset()
                
                # Clear civilians (disaster is reset)
                if hasattr(env, 'civilians'):
                    env.civilians = []
                
                # Reset session state
                st.session_state.simulation_data['steps'] = []
                st.session_state.disaster_triggered = False
                
                st.success("🔄 Simulation reset! Civilians cleared. Ready for new disaster.")
                logger.info("Simulation reset - civilians cleared")
            except Exception as e:
                self.handle_error(f"Reset error: {e}")
                st.error("❌ Failed to reset simulation")
        else:
            st.warning("⚠️ No active simulation to reset")
    
    def trigger_disaster(self):
        """Trigger disaster in the simulation - THIS IS WHERE CIVILIANS ARE CREATED"""
        # Check if user has permission
        if st.session_state.user_role != "admin":
            st.error("❌ Admin permission required to trigger disaster")
            return
        
        if st.session_state.environment and not st.session_state.disaster_triggered:
            try:
                env = st.session_state.environment
                
                # Get the civilian count from session state
                num_civilians = st.session_state.get('num_civilians', 8)
                
                # Initialize civilians list if it doesn't exist
                if not hasattr(env, 'civilians'):
                    env.civilians = []
                
                # Clear any existing civilians (should be empty, but just in case)
                env.civilians = []
                
                # Generate civilians at the moment of disaster
                logger.info(f"Creating {num_civilians} civilians due to disaster")
                
                for i in range(num_civilians):
                    # Find valid position (not in a building)
                    attempts = 0
                    while attempts < 50:
                        x = np.random.randint(0, env.grid_size)
                        y = np.random.randint(0, env.grid_size)
                        
                        # Check if position is valid
                        if hasattr(env, 'grid') and y < env.grid_size and x < env.grid_size:
                            cell_value = env.grid[y, x]
                            # Valid positions: not a building (assuming 2 = building)
                            if cell_value != 2:
                                break
                        else:
                            break  # Use the position if we can't validate
                        
                        attempts += 1
                    
                    # Add civilian
                    env.civilians.append({
                        'id': f'civilian_{i}',
                        'position': np.array([x, y]),
                        'rescued': False,
                        'health': np.random.randint(30, 100),  # Some are more injured than others
                        'found': False,
                        'trapped_in_building': np.random.choice([True, False], p=[0.3, 0.7])  # 30% trapped
                    })
                
                # Also trigger other disaster effects from the environment
                # (This depends on your environment's trigger_disaster method)
                if hasattr(env, 'trigger_disaster'):
                    env.trigger_disaster()
                
                # Update session state
                st.session_state.disaster_triggered = True
                st.session_state.simulation_data['disaster_time'] = datetime.now()
                
                st.success(f"🚨 DISASTER TRIGGERED! {num_civilians} civilians need rescue!")
                st.warning("⚠️ Agents: Locate and rescue civilians!")
                logger.info(f"Disaster triggered with {len(env.civilians)} civilians")
                st.rerun()
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
            # Check if user has permission
            if st.session_state.user_role != "admin":
                st.error("❌ Admin permission required to save simulation")
                return
                
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
            # Check if user is authenticated
            if not st.session_state.authenticated:
                self.render_login()
            else:
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