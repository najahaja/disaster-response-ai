#!/usr/bin/env python3
"""
Script to add Legend import and integration to dashboard/app.py
"""

# Read the file
with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and modify the import section (add Legend import after SimulationViewer)
for i, line in enumerate(lines):
    if 'from dashboard.components.simulation_viewer import SimulationViewer' in line:
        # Insert Legend import after this line
        lines.insert(i + 1, '    from dashboard.components.legend import Legend\n')
        print(f"Added Legend import at line {i+2}")
        break

# Find and modify __init__ method (add self.legend = Legend())
for i, line in enumerate(lines):
    if line.strip() == 'self.initialize_components()':
        # Add legend initialization after this line
        lines.insert(i + 1, '        self.legend = Legend()\n')
        print(f"Added Legend initialization at line {i+2}")
        break

# Find and modify render_enhanced_simulation_view (add self.legend.render())
for i, line in enumerate(lines):
    if 'st.subheader("📊 Real‑time Metrics")' in line or 'st.subheader("📊 Real-time Metrics")' in line:
        # Find the next line with self.render_realtime_metrics()
        for j in range(i, min(i+5, len(lines))):
            if 'self.render_realtime_metrics()' in lines[j]:
                # Add legend rendering after this line
                lines.insert(j + 1, '        \n')
                lines.insert(j + 2, '        # Render legend below map and controls\n')
                lines.insert(j + 3, '        self.legend.render()\n')
                print(f"Added Legend rendering at line {j+4}")
                break
        break

# Write the modified file
with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Successfully modified dashboard/app.py")
