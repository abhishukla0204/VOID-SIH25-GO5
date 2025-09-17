#!/usr/bin/env python3
"""
Rockfall Detection Dashboard
===========================

Interactive Streamlit dashboard for real-time monitoring and visualization
of the rockfall detection and prediction system.

Features:
- Real-time video detection monitoring
- Sensor data visualization and alerts
- DEM risk analysis display
- Historical alert tracking
- System status monitoring
- Configuration management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import threading
import queue
from PIL import Image
import io
import base64

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from detection.realtime_detector import RockfallDetector
from sensors.sensor_alerts import SensorDataProcessor
from dem_analysis.dem_processor import DEMAnalyzer


class RockfallDashboard:
    """Main dashboard class for the rockfall monitoring system"""
    
    def __init__(self):
        """Initialize dashboard"""
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Rockfall Detection System",
            page_icon="🏔️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'detection_active' not in st.session_state:
            st.session_state.detection_active = False
        if 'sensor_monitoring' not in st.session_state:
            st.session_state.sensor_monitoring = False
        if 'alerts_data' not in st.session_state:
            st.session_state.alerts_data = []
        if 'video_detector' not in st.session_state:
            st.session_state.video_detector = None
        if 'sensor_processor' not in st.session_state:
            st.session_state.sensor_processor = None
        if 'dem_analyzer' not in st.session_state:
            st.session_state.dem_analyzer = None
    
    def render_header(self):
        """Render dashboard header"""
        st.title("🏔️ AI-Based Rockfall Detection & Prediction System")
        st.markdown("Real-time monitoring and analysis for rockfall hazard assessment")
        
        # System status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "🟢 Active" if st.session_state.detection_active else "🔴 Inactive"
            st.metric("Video Detection", status)
        
        with col2:
            status = "🟢 Monitoring" if st.session_state.sensor_monitoring else "🔴 Offline"
            st.metric("Sensor System", status)
        
        with col3:
            alert_count = len(st.session_state.alerts_data)
            st.metric("Active Alerts", alert_count)
        
        with col4:
            st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("🎛️ System Controls")
        
        # Detection system controls
        st.sidebar.subheader("🎥 Video Detection")
        
        if st.sidebar.button("Start Detection" if not st.session_state.detection_active else "Stop Detection"):
            st.session_state.detection_active = not st.session_state.detection_active
            if st.session_state.detection_active:
                self.start_video_detection()
            else:
                self.stop_video_detection()
        
        # Detection settings
        if st.session_state.detection_active:
            confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
            alert_threshold = st.sidebar.slider("Alert Threshold", 1, 10, 3)
        
        st.sidebar.subheader("📊 Sensor Monitoring")
        
        if st.sidebar.button("Start Monitoring" if not st.session_state.sensor_monitoring else "Stop Monitoring"):
            st.session_state.sensor_monitoring = not st.session_state.sensor_monitoring
            if st.session_state.sensor_monitoring:
                self.start_sensor_monitoring()
            else:
                self.stop_sensor_monitoring()
        
        # DEM Analysis
        st.sidebar.subheader("🗺️ DEM Analysis")
        
        dem_files = list(Path("data/DEM").glob("*.tif")) if Path("data/DEM").exists() else []
        if dem_files:
            selected_dem = st.sidebar.selectbox(
                "Select DEM File",
                options=[f.name for f in dem_files]
            )
            
            if st.sidebar.button("Analyze DEM"):
                self.analyze_dem(f"data/DEM/{selected_dem}")
        
        # System settings
        st.sidebar.subheader("⚙️ Settings")
        auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=True)
        
        if auto_refresh:
            time.sleep(5)
            st.rerun()
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Dashboard"):
            st.rerun()
    
    def start_video_detection(self):
        """Start video detection system"""
        try:
            st.session_state.video_detector = RockfallDetector(
                confidence=0.5,
                device='cpu',
                alert_threshold=3
            )
            st.sidebar.success("Video detection started!")
        except Exception as e:
            st.sidebar.error(f"Failed to start detection: {e}")
    
    def stop_video_detection(self):
        """Stop video detection system"""
        st.session_state.video_detector = None
        st.sidebar.info("Video detection stopped")
    
    def start_sensor_monitoring(self):
        """Start sensor monitoring system"""
        try:
            st.session_state.sensor_processor = SensorDataProcessor()
            st.sidebar.success("Sensor monitoring started!")
        except Exception as e:
            st.sidebar.error(f"Failed to start monitoring: {e}")
    
    def stop_sensor_monitoring(self):
        """Stop sensor monitoring system"""
        st.session_state.sensor_processor = None
        st.sidebar.info("Sensor monitoring stopped")
    
    def analyze_dem(self, dem_path: str):
        """Analyze DEM file"""
        try:
            with st.spinner("Analyzing DEM data..."):
                st.session_state.dem_analyzer = DEMAnalyzer(dem_path)
                st.sidebar.success("DEM analysis completed!")
        except Exception as e:
            st.sidebar.error(f"DEM analysis failed: {e}")
    
    def render_video_detection_tab(self):
        """Render video detection monitoring tab"""
        st.subheader("🎥 Real-time Video Detection")
        
        if not st.session_state.detection_active:
            st.info("Video detection is not active. Use the sidebar to start detection.")
            return
        
        # Video feed placeholder
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Live Video Feed")
            video_placeholder = st.empty()
            
            # Simulate video detection (in real implementation, this would show actual video)
            with video_placeholder.container():
                st.info("🎥 Video detection active - Live feed would be displayed here")
                st.text("Note: In a real deployment, this would show the live video stream")
                st.text("with real-time bounding boxes around detected rocks.")
        
        with col2:
            st.subheader("Detection Statistics")
            
            # Mock detection statistics
            total_detections = np.random.randint(0, 50)
            confidence_avg = np.random.uniform(0.6, 0.9)
            fps = np.random.uniform(15, 25)
            
            st.metric("Total Detections", total_detections)
            st.metric("Avg Confidence", f"{confidence_avg:.2f}")
            st.metric("Processing FPS", f"{fps:.1f}")
            
            # Recent detections
            st.subheader("Recent Detections")
            if total_detections > 0:
                detection_times = [
                    datetime.now() - timedelta(minutes=np.random.randint(0, 60))
                    for _ in range(min(5, total_detections))
                ]
                
                for i, dt in enumerate(detection_times):
                    confidence = np.random.uniform(0.5, 0.95)
                    st.text(f"#{i+1}: {dt.strftime('%H:%M:%S')} (conf: {confidence:.2f})")
            else:
                st.text("No recent detections")
    
    def render_sensor_monitoring_tab(self):
        """Render sensor monitoring tab"""
        st.subheader("📊 Sensor Data Monitoring")
        
        if not st.session_state.sensor_monitoring:
            st.info("Sensor monitoring is not active. Use the sidebar to start monitoring.")
            return
        
        # Generate real-time sensor data
        if st.session_state.sensor_processor:
            # Generate synthetic data for demo
            sensor_data = st.session_state.sensor_processor.generate_synthetic_sensor_data(
                duration_hours=1, interval_minutes=5
            )
            
            # Analyze current data
            analysis_result = st.session_state.sensor_processor.analyze_sensor_data(sensor_data)
            
            # Display current readings
            col1, col2, col3, col4 = st.columns(4)
            
            latest_data = sensor_data.iloc[-1]
            
            with col1:
                st.metric(
                    "Vibration", 
                    f"{latest_data['vibration_velocity']:.2f} mm/s",
                    delta=f"{np.random.uniform(-0.1, 0.1):.2f}"
                )
            
            with col2:
                st.metric(
                    "Temperature", 
                    f"{latest_data['temperature']:.1f}°C",
                    delta=f"{np.random.uniform(-1, 1):.1f}"
                )
            
            with col3:
                st.metric(
                    "Humidity", 
                    f"{latest_data['humidity']:.1f}%",
                    delta=f"{np.random.uniform(-2, 2):.1f}"
                )
            
            with col4:
                risk_color = "red" if analysis_result['risk_level'] == 'HIGH' else "orange" if analysis_result['risk_level'] == 'MEDIUM' else "green"
                st.metric(
                    "Risk Level", 
                    analysis_result['risk_level'],
                    delta=f"Score: {analysis_result['risk_score']:.3f}"
                )
            
            # Time series plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Vibration & Acceleration")
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Vibration Velocity", "Acceleration"),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=sensor_data['timestamp'],
                        y=sensor_data['vibration_velocity'],
                        name="Vibration (mm/s)",
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=sensor_data['timestamp'],
                        y=sensor_data['acceleration'],
                        name="Acceleration (m/s²)",
                        line=dict(color='orange')
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Environmental Conditions")
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Temperature & Humidity", "Precipitation"),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=sensor_data['timestamp'],
                        y=sensor_data['temperature'],
                        name="Temperature (°C)",
                        line=dict(color='blue'),
                        yaxis='y1'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=sensor_data['timestamp'],
                        y=sensor_data['precipitation'],
                        name="Precipitation (mm/h)",
                        marker_color='lightblue'
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Alert information
            if analysis_result['alerts']:
                st.subheader("🚨 Active Sensor Alerts")
                
                for alert in analysis_result['alerts']:
                    alert_color = {
                        'HIGH': '#ff4444',
                        'MEDIUM': '#ff8800', 
                        'LOW': '#ffcc00'
                    }.get(alert['level'], '#888888')
                    
                    st.markdown(
                        f"<div style='padding: 10px; background-color: {alert_color}20; "
                        f"border-left: 5px solid {alert_color}; margin: 5px 0;'>"
                        f"<strong>{alert['level']} ALERT:</strong> {alert['message']}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
    
    def render_dem_analysis_tab(self):
        """Render DEM analysis tab"""
        st.subheader("🗺️ DEM Risk Analysis")
        
        if not st.session_state.dem_analyzer:
            st.info("No DEM analysis loaded. Use the sidebar to select and analyze a DEM file.")
            return
        
        analyzer = st.session_state.dem_analyzer
        
        # Perform risk assessment
        with st.spinner("Computing risk analysis..."):
            risk_results = analyzer.assess_rockfall_risk()
            critical_zones = analyzer.identify_critical_zones(risk_results)
        
        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_pixels = np.sum(~np.isnan(risk_results['combined_risk']))
        high_risk_pixels = np.sum(risk_results['risk_classification'] == 3)
        medium_risk_pixels = np.sum(risk_results['risk_classification'] == 2)
        
        with col1:
            st.metric("High Risk Areas", f"{high_risk_pixels/total_pixels*100:.1f}%")
        
        with col2:
            st.metric("Medium Risk Areas", f"{medium_risk_pixels/total_pixels*100:.1f}%")
        
        with col3:
            st.metric("Critical Zones", len(critical_zones))
        
        with col4:
            max_slope = float(np.nanmax(risk_results['slope']))
            st.metric("Max Slope", f"{max_slope:.1f}°")
        
        # Display risk maps
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Elevation Map")
            fig = px.imshow(
                analyzer.dem_data,
                color_continuous_scale='terrain',
                title="Digital Elevation Model"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Slope Analysis")
            fig = px.imshow(
                risk_results['slope'],
                color_continuous_scale='Reds',
                title="Slope (degrees)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk classification map
        st.subheader("Risk Classification")
        
        # Create custom colorscale for risk levels
        risk_colorscale = [
            [0, 'green'],    # Safe
            [0.33, 'yellow'], # Low risk
            [0.66, 'orange'], # Medium risk
            [1, 'red']        # High risk
        ]
        
        fig = px.imshow(
            risk_results['risk_classification'],
            color_continuous_scale=risk_colorscale,
            title="Rockfall Risk Classification"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Critical zones information
        if critical_zones:
            st.subheader("🎯 Critical Zones")
            
            zones_df = pd.DataFrame([
                {
                    'Zone ID': zone['zone_id'],
                    'Area (pixels)': zone['area_pixels'],
                    'Mean Slope (°)': f"{zone['statistics']['mean_slope']:.1f}",
                    'Max Slope (°)': f"{zone['statistics']['max_slope']:.1f}",
                    'Priority': zone['priority']
                }
                for zone in critical_zones[:10]  # Show top 10
            ])
            
            st.dataframe(zones_df, use_container_width=True)
    
    def render_alerts_tab(self):
        """Render alerts and history tab"""
        st.subheader("🚨 Alert Management")
        
        # Alert summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Alerts Today", len(st.session_state.alerts_data))
        
        with col2:
            high_priority = len([a for a in st.session_state.alerts_data if a.get('level') == 'HIGH'])
            st.metric("High Priority", high_priority)
        
        with col3:
            recent_alerts = len([
                a for a in st.session_state.alerts_data 
                if datetime.fromisoformat(a.get('timestamp', '2000-01-01')) > datetime.now() - timedelta(hours=1)
            ])
            st.metric("Last Hour", recent_alerts)
        
        # Alert timeline
        if st.session_state.alerts_data:
            st.subheader("Alert Timeline")
            
            # Create timeline chart
            alert_df = pd.DataFrame(st.session_state.alerts_data)
            if not alert_df.empty:
                fig = px.timeline(
                    alert_df,
                    x_start="timestamp",
                    x_end="timestamp",
                    y="type",
                    color="level",
                    title="Alert Timeline"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent alerts table
        st.subheader("Recent Alerts")
        
        if st.session_state.alerts_data:
            recent_alerts_df = pd.DataFrame(st.session_state.alerts_data[-20:])  # Last 20 alerts
            st.dataframe(recent_alerts_df, use_container_width=True)
        else:
            st.info("No alerts recorded yet.")
        
        # Alert configuration
        st.subheader("Alert Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Email Notifications", placeholder="admin@example.com")
            st.selectbox("Alert Frequency", ["Immediate", "Every 5 minutes", "Hourly"])
        
        with col2:
            st.multiselect("Alert Types", 
                          ["Video Detection", "Sensor Alerts", "Risk Threshold", "System Status"])
            st.checkbox("Send SMS Alerts")
    
    def run(self):
        """Run the main dashboard"""
        self.render_header()
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎥 Video Detection", 
            "📊 Sensor Monitoring", 
            "🗺️ DEM Analysis", 
            "🚨 Alerts"
        ])
        
        with tab1:
            self.render_video_detection_tab()
        
        with tab2:
            self.render_sensor_monitoring_tab()
        
        with tab3:
            self.render_dem_analysis_tab()
        
        with tab4:
            self.render_alerts_tab()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666666;'>"
            "Rockfall Detection System v1.0 | "
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            "</div>",
            unsafe_allow_html=True
        )


def main():
    """Main function to run the dashboard"""
    dashboard = RockfallDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()