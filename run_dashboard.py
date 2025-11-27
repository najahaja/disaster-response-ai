#!/usr/bin/env python3
"""
Main script to run the Disaster Response AI Dashboard
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dashboard.app import DisasterResponseDashboard

def main():
    """Run the dashboard application"""
    try:
        dashboard = DisasterResponseDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Dashboard error: {e}")
        st.info("Please make sure all dependencies are installed and the project structure is correct.")

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx():
            main()
        else:
            import sys
            from streamlit.web import cli as stcli
            sys.argv = ["streamlit", "run", sys.argv[0]]
            sys.exit(stcli.main())
    except ImportError:
        # Fallback for older versions or if something goes wrong
        main()