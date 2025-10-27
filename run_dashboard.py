#!/usr/bin/env python3
"""
Main entry point for the Streamlit dashboard
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dashboard.app import DisasterResponseDashboard

def main():
    """Main function to run the dashboard"""
    print("🚀 Starting Disaster Response AI Dashboard...")
    print("🌐 Opening web interface at http://localhost:8501")
    
    # Note: Streamlit runs differently than normal Python scripts
    # This file is meant to be run with: streamlit run run_dashboard.py
    
    # For direct execution, we'll import and run the dashboard
    try:
        dashboard = DisasterResponseDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Error starting dashboard: {e}")
        print(f"❌ Dashboard error: {e}")

if __name__ == "__main__":
    # Check if running with streamlit
    if 'streamlit' in sys.modules:
        main()
    else:
        print("🔧 To run the dashboard, use:")
        print("   streamlit run run_dashboard.py")
        print("\n📁 Alternatively, run the components directly:")
        print("   python run_training.py  # For training")
        print("   python test_visualization.py  # For visualization tests")