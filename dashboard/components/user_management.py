import streamlit as st
import pandas as pd
from datetime import datetime

class UserManagement:
    """Component for user management interface"""
    
    def __init__(self, auth_manager):
        self.auth = auth_manager
    
    def render_user_list(self):
        """Render user list with management options"""
        users = self.auth.db.get_all_users()
        
        if not users:
            st.info("No users found")
            return
        
        # Create DataFrame
        user_data = []
        for user in users:
            user_data.append({
                'ID': user['id'],
                'Username': user['username'],
                'Email': user['email'],
                'Role': user['role'].upper(),
                'Status': '✅ Active' if user['is_active'] else '❌ Inactive',
                'Created': user['created_at'],
                'Last Login': user['last_login'] or 'Never',
                'Failed Attempts': user['failed_attempts']
            })
        
        df = pd.DataFrame(user_data)
        
        # Display table
        st.dataframe(
            df,
            column_config={
                "ID": st.column_config.NumberColumn(width="small"),
                "Status": st.column_config.TextColumn(width="small"),
                "Failed Attempts": st.column_config.NumberColumn(width="small"),
            },
            use_container_width=True,
            hide_index=True
        )
    
    def render_user_actions(self):
        """Render user action controls"""
        st.subheader("User Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.form("change_role_form"):
                user_id = st.number_input("User ID", min_value=1)
                new_role = st.selectbox("New Role", ["viewer", "admin"])
                
                if st.form_submit_button("Change Role"):
                    if self.auth.db.update_user_role(user_id, new_role, 
                                                    self.auth.get_current_user()['id']):
                        st.success(f"✅ Role updated for user {user_id}")
                        st.rerun()
        
        with col2:
            with st.form("reset_password_form"):
                st.text_input("User ID", key="reset_user_id")
                new_password = st.text_input("New Password", type="password")
                
                if st.form_submit_button("Reset Password"):
                    st.info("Feature coming soon")
        
        with col3:
            with st.form("toggle_status_form"):
                toggle_user_id = st.number_input("User ID", min_value=1, key="toggle_id")
                
                if st.form_submit_button("Toggle Status"):
                    st.info("Feature coming soon")