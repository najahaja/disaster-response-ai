#!/usr/bin/env python3
"""
Dashboard Component Tests for Week 6
Tests Streamlit dashboard components and functionality
"""

import sys
import os
import unittest
import numpy as np
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDashboardComponents(unittest.TestCase):
    """
    Test dashboard components and functionality
    """
    
    def setUp(self):
        """Set up test configuration"""
        self.config = {
            'environment': {
                'grid_size': 20,
                'cell_size': 30,
                'max_steps': 100,
                'cell_types': {
                    'ROAD': 0,
                    'BUILDING': 1,
                    'HOSPITAL': 2
                }
            },
            'visualization': {
                'colors': {
                    0: (100, 100, 100),
                    1: (139, 69, 19),
                    2: (255, 0, 0)
                }
            }
        }
    
    def test_control_panel_creation(self):
        """Test control panel component creation"""
        print("🎮 Testing control panel creation...")
        
        try:
            from dashboard.components.controls import ControlPanel
            
            panel = ControlPanel()
            self.assertIsNotNone(panel)
            
            # Test configuration methods
            config = panel.get_current_configuration()
            self.assertIsInstance(config, dict)
            self.assertIn('environment', config)
            self.assertIn('agents', config)
            self.assertIn('settings', config)
            
            # Test validation
            errors = panel.validate_configuration(config)
            self.assertIsInstance(errors, list)
            
            print("✅ Control panel creation test passed")
            
        except ImportError as e:
            self.skipTest(f"Control panel not available: {e}")
    
    def test_metrics_display_functionality(self):
        """Test metrics display component functionality"""
        print("📊 Testing metrics display functionality...")
        
        try:
            from dashboard.components.metrics_display import MetricsDisplay
            
            metrics = MetricsDisplay()
            self.assertIsNotNone(metrics)
            
            # Create mock environment
            mock_env = type('MockEnv', (), {
                'civilians': [{'rescued': True}, {'rescued': False}],
                'agents': {'drone_1': type('MockAgent', (), {'steps_taken': 10})()},
                'step_count': 15,
                'collapsed_buildings': [(1, 1)],
                'blocked_roads': [(2, 2)]
            })()
            
            # Test metrics update
            metrics.update_metrics(mock_env, {})
            
            # Test metrics export
            exported = metrics.export_metrics()
            self.assertIsInstance(exported, dict)
            self.assertIn('metrics_history', exported)
            
            print("✅ Metrics display functionality test passed")
            
        except ImportError as e:
            self.skipTest(f"Metrics display not available: {e}")
    
    def test_simulation_viewer_creation(self):
        """Test simulation viewer component creation"""
        print("👁️ Testing simulation viewer creation...")
        
        try:
            from dashboard.components.simulation_viewer import SimulationViewer
            
            viewer = SimulationViewer()
            self.assertIsNotNone(viewer)
            
            # Create mock environment
            mock_env = type('MockEnv', (), {
                'grid': np.random.randint(0, 3, (20, 20)),
                'agents': {},
                'civilians': [],
                'collapsed_buildings': [],
                'blocked_roads': [],
                'grid_size': 20
            })()
            
            # Test plot creation
            fig = viewer.create_simulation_plot(mock_env)
            self.assertIsNotNone(fig)
            
            # Test performance chart
            perf_data = {'steps': [{'rescues': i, 'efficiency': i*2} for i in range(5)]}
            perf_fig = viewer.create_performance_chart(perf_data)
            self.assertIsNotNone(perf_fig)
            
            print("✅ Simulation viewer creation test passed")
            
        except ImportError as e:
            self.skipTest(f"Simulation viewer not available: {e}")
    
    def test_dashboard_integration(self):
        """Test dashboard component integration"""
        print("🔄 Testing dashboard component integration...")
        
        try:
            from dashboard.components.controls import ControlPanel
            from dashboard.components.metrics_display import MetricsDisplay
            from dashboard.components.simulation_viewer import SimulationViewer
            
            # Create all components
            controls = ControlPanel()
            metrics = MetricsDisplay()
            viewer = SimulationViewer()
            
            # Test they can work together
            config = controls.get_current_configuration()
            self.assertIsInstance(config, dict)
            
            # Create mock data for metrics
            mock_env = type('MockEnv', (), {
                'civilians': [{'rescued': False}],
                'agents': {},
                'step_count': 1,
                'collapsed_buildings': [],
                'blocked_roads': []
            })()
            
            metrics.update_metrics(mock_env, {})
            
            # Test viewer with minimal data
            minimal_env = type('MockEnv', (), {
                'grid': np.zeros((10, 10)),
                'agents': {},
                'civilians': [],
                'collapsed_buildings': [],
                'blocked_roads': [],
                'grid_size': 10
            })()
            
            fig = viewer.create_simulation_plot(minimal_env)
            self.assertIsNotNone(fig)
            
            print("✅ Dashboard component integration test passed")
            
        except ImportError as e:
            self.skipTest(f"Dashboard components not available: {e}")

def run_dashboard_tests():
    """Run all dashboard tests"""
    print("🧪 Dashboard Component Tests - Week 6")
    print("=" * 50)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDashboardComponents))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    print("📊 Dashboard Test Summary")
    print("=" * 50)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("🎉 All dashboard tests passed!")
        return True
    else:
        print("⚠️ Some dashboard tests failed or were skipped")
        return False

if __name__ == "__main__":
    success = run_dashboard_tests()
    sys.exit(0 if success else 1)