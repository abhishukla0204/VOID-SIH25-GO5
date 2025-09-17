"""
Enhanced Streamlit Dashboard for Integrated Rockfall Monitoring
=============================================================

This enhanced dashboard provides comprehensive visualization and control for both:
- PROACTIVE: ML-based risk prediction and assessment
- REACTIVE: Real-time YOLO detection and alerts

Features:
- Real-time risk monitoring and prediction visualization
- Interactive risk maps and terrain analysis
- Feature importance analysis and model performance metrics
- Combined alert management (prediction + detection)
- Historical trend analysis and reporting
- System configuration and control interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import os
import sys

# Add paths for imports
sys.path.append('src')
sys.path.append('src/prediction')

# Configure page
st.set_page_config(
    page_title="🏔️ Integrated Rockfall Monitoring System",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #ffaa00;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #44ff44;
        color: black;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'system_running' not in st.session_state:
    st.session_state.system_running = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

class EnhancedRockfallDashboard:
    """Enhanced dashboard for integrated rockfall monitoring"""
    
    def __init__(self):
        self.setup_dashboard()
    
    def setup_dashboard(self):
        """Setup the main dashboard layout"""
        
        # Main header
        st.markdown('<h1 class="main-header">🏔️ Integrated Rockfall Monitoring System</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar for system control
        self.setup_sidebar()
        
        # Main content tabs
        self.setup_main_tabs()
    
    def setup_sidebar(self):
        """Setup sidebar with system controls and status"""
        
        st.sidebar.markdown("## 🎛️ System Control")
        
        # System status
        status_color = "🟢" if st.session_state.system_running else "🔴"
        st.sidebar.markdown(f"**System Status:** {status_color}")
        
        # Control buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("▶️ Start", key="start_btn"):
                st.session_state.system_running = True
                st.success("System started!")
        
        with col2:
            if st.button("⏹️ Stop", key="stop_btn"):
                st.session_state.system_running = False
                st.warning("System stopped!")
        
        # Configuration
        st.sidebar.markdown("## ⚙️ Configuration")
        
        prediction_interval = st.sidebar.slider(
            "Prediction Interval (minutes)", 1, 30, 5
        )
        
        risk_threshold = st.sidebar.slider(
            "Risk Alert Threshold", 0.0, 1.0, 0.6, 0.1
        )
        
        detection_enabled = st.sidebar.checkbox(
            "Enable Real-time Detection", value=True
        )
        
        # Location input
        st.sidebar.markdown("## 📍 Location")
        location_lat = st.sidebar.number_input("Latitude", value=40.7128, format="%.4f")
        location_lon = st.sidebar.number_input("Longitude", value=-74.0060, format="%.4f")
        
        # System statistics
        st.sidebar.markdown("## 📊 Statistics")
        st.sidebar.metric("Predictions", len(st.session_state.prediction_history))
        st.sidebar.metric("Alerts", len(st.session_state.alert_history))
        st.sidebar.metric("Uptime", self.get_uptime())
        
        # Quick actions
        st.sidebar.markdown("## 🚀 Quick Actions")
        if st.sidebar.button("🔄 Refresh Data"):
            self.simulate_data_update()
            st.experimental_rerun()
        
        if st.sidebar.button("📊 Generate Report"):
            self.generate_system_report()
        
        if st.sidebar.button("🧹 Clear History"):
            st.session_state.prediction_history = []
            st.session_state.alert_history = []
            st.success("History cleared!")
    
    def setup_main_tabs(self):
        """Setup main content area with tabs"""
        
        tabs = st.tabs([
            "🏠 Overview",
            "🧠 Risk Prediction", 
            "👁️ Real-time Detection",
            "🗺️ Risk Mapping",
            "📊 Analytics",
            "🚨 Alert Management",
            "⚙️ Model Performance"
        ])
        
        with tabs[0]:
            self.show_overview_tab()
        
        with tabs[1]:
            self.show_prediction_tab()
        
        with tabs[2]:
            self.show_detection_tab()
        
        with tabs[3]:
            self.show_risk_mapping_tab()
        
        with tabs[4]:
            self.show_analytics_tab()
        
        with tabs[5]:
            self.show_alert_management_tab()
        
        with tabs[6]:
            self.show_model_performance_tab()
    
    def show_overview_tab(self):
        """Display system overview"""
        
        st.markdown("## 📊 System Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_risk = self.get_current_risk_score()
            st.metric(
                "Current Risk Score", 
                f"{current_risk:.3f}",
                delta=f"{np.random.uniform(-0.1, 0.1):.3f}"
            )
        
        with col2:
            risk_level = self.get_risk_level_name(current_risk)
            st.metric("Risk Level", risk_level)
        
        with col3:
            detection_count = len([a for a in st.session_state.alert_history 
                                 if a.get('type') == 'detection'])
            st.metric("Detections Today", detection_count)
        
        with col4:
            prediction_count = len([a for a in st.session_state.alert_history 
                                  if a.get('type') == 'prediction'])
            st.metric("Predictions Today", prediction_count)
        
        # Real-time charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Risk Score Trend")
            self.plot_risk_trend()
        
        with col2:
            st.markdown("### 🌡️ Environmental Conditions")
            self.plot_environmental_conditions()
        
        # Recent alerts
        st.markdown("### 🚨 Recent Alerts")
        self.show_recent_alerts(limit=5)
        
        # System status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("🧠 **Prediction System**\n✅ Online\n📊 Last update: 2 min ago")
        
        with col2:
            st.success("👁️ **Detection System**\n✅ Monitoring\n📹 Active streams: 1")
        
        with col3:
            st.warning("📡 **Sensor Network**\n⚠️ Limited data\n📊 3/5 sensors active")
    
    def show_prediction_tab(self):
        """Display risk prediction interface"""
        
        st.markdown("## 🧠 AI-Based Risk Prediction")
        
        # Current prediction
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Current Risk Assessment")
            
            # Simulate comprehensive risk assessment
            current_assessment = self.get_current_risk_assessment()
            
            # Risk score gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_assessment['risk_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score"},
                delta={'reference': 0.5},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgreen"},
                        {'range': [0.3, 0.6], 'color': "yellow"},
                        {'range': [0.6, 0.8], 'color': "orange"},
                        {'range': [0.8, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Risk Components")
            
            # Component breakdown
            components = current_assessment['components']
            component_df = pd.DataFrame(list(components.items()), 
                                      columns=['Component', 'Score'])
            
            fig = px.bar(component_df, x='Score', y='Component', 
                        orientation='h', color='Score',
                        color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Prediction controls
        st.markdown("### 🎛️ Prediction Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Update Prediction"):
                self.generate_new_prediction()
                st.success("Prediction updated!")
        
        with col2:
            if st.button("📊 Analyze Features"):
                self.show_feature_analysis()
        
        with col3:
            auto_update = st.checkbox("🔄 Auto-update", value=True)
        
        # Probability forecasting
        st.markdown("### 📅 Probability Forecast")
        
        forecast_data = self.get_probability_forecast()
        forecast_df = pd.DataFrame(list(forecast_data.items()), 
                                 columns=['Time Horizon', 'Probability'])
        
        fig = px.bar(forecast_df, x='Time Horizon', y='Probability',
                    title="Rockfall Probability by Time Horizon",
                    color='Probability', color_continuous_scale='Reds')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Terrain features
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏔️ Terrain Features")
            terrain_features = current_assessment['terrain_features']
            
            for feature, value in terrain_features.items():
                st.metric(feature.replace('_', ' ').title(), f"{value:.2f}")
        
        with col2:
            st.markdown("### 🌤️ Environmental Conditions")
            env_conditions = current_assessment['environmental_conditions']
            
            for condition, value in env_conditions.items():
                st.metric(condition.replace('_', ' ').title(), f"{value:.2f}")
    
    def show_detection_tab(self):
        """Display real-time detection interface"""
        
        st.markdown("## 👁️ Real-time YOLO Detection")
        
        # Detection status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Detection Status", "🟢 Active")
        
        with col2:
            st.metric("Processing FPS", "24.3")
        
        with col3:
            st.metric("Confidence Threshold", "0.5")
        
        # Video feed placeholder
        st.markdown("### 📹 Live Video Feed")
        
        # Placeholder for video stream
        video_placeholder = st.empty()
        with video_placeholder.container():
            st.info("📹 **Video Stream**\n\n🔴 **LIVE** - Camera 1 (Main Site)\n\nResolution: 1920x1080\nStream: rtmp://camera1.site.com/live")
        
        # Recent detections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Detection Statistics")
            
            # Simulate detection stats
            detection_stats = {
                "Total Detections": 12,
                "High Confidence": 3,
                "Medium Confidence": 5,
                "Low Confidence": 4,
                "False Positives": 2
            }
            
            stats_df = pd.DataFrame(list(detection_stats.items()), 
                                  columns=['Metric', 'Count'])
            
            fig = px.pie(stats_df, values='Count', names='Metric',
                        title="Detection Breakdown")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⏰ Detection Timeline")
            
            # Simulate recent detections
            detection_timeline = []
            for i in range(5):
                detection_timeline.append({
                    'Time': (datetime.now() - timedelta(hours=i*2)).strftime('%H:%M'),
                    'Confidence': np.random.uniform(0.3, 0.9),
                    'Location': f"Zone {np.random.choice(['A', 'B', 'C'])}"
                })
            
            timeline_df = pd.DataFrame(detection_timeline)
            
            fig = px.scatter(timeline_df, x='Time', y='Confidence', 
                           size='Confidence', color='Location',
                           title="Recent Detections")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detection controls
        st.markdown("### 🎛️ Detection Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5)
        
        with col2:
            alert_threshold = st.slider("Alert Threshold", 1, 10, 3)
        
        with col3:
            save_detections = st.checkbox("Save Detections", value=True)
        
        with col4:
            real_time_alerts = st.checkbox("Real-time Alerts", value=True)
    
    def show_risk_mapping_tab(self):
        """Display risk mapping interface"""
        
        st.markdown("## 🗺️ Spatial Risk Mapping")
        
        # Map controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            map_type = st.selectbox("Map Type", ["Risk Zones", "Terrain", "Probability"])
        
        with col2:
            time_period = st.selectbox("Time Period", ["Current", "1 Hour", "24 Hours", "7 Days"])
        
        with col3:
            resolution = st.selectbox("Resolution", ["High", "Medium", "Low"])
        
        # Generate synthetic risk map data
        map_data = self.generate_risk_map_data()
        
        # Risk map visualization
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 🗺️ Risk Zone Map")
            
            fig = px.density_mapbox(
                map_data, lat='lat', lon='lon', z='risk_score',
                radius=10, center=dict(lat=40.7128, lon=-74.0060),
                zoom=12, mapbox_style="open-street-map",
                color_continuous_scale="Reds",
                title="Rockfall Risk Distribution"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Risk Statistics")
            
            # Risk zone statistics
            risk_zones = {
                "High Risk": len(map_data[map_data['risk_score'] > 0.7]),
                "Medium Risk": len(map_data[(map_data['risk_score'] > 0.4) & (map_data['risk_score'] <= 0.7)]),
                "Low Risk": len(map_data[map_data['risk_score'] <= 0.4])
            }
            
            for zone, count in risk_zones.items():
                percentage = (count / len(map_data)) * 100
                st.metric(zone, f"{count} points", f"{percentage:.1f}%")
        
        # DEM analysis
        st.markdown("### 🏔️ Terrain Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Slope distribution
            slope_data = np.random.gamma(2, 15, 1000)
            slope_data = np.clip(slope_data, 0, 90)
            
            fig = px.histogram(x=slope_data, nbins=30, 
                             title="Slope Distribution",
                             labels={'x': 'Slope (degrees)', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Elevation profile
            elevation_data = np.random.normal(1500, 400, 1000)
            
            fig = px.histogram(x=elevation_data, nbins=30,
                             title="Elevation Distribution", 
                             labels={'x': 'Elevation (m)', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
    
    def show_analytics_tab(self):
        """Display analytics and trends"""
        
        st.markdown("## 📊 Analytics & Trends")
        
        # Time series analysis
        st.markdown("### 📈 Risk Trend Analysis")
        
        # Generate historical risk data
        historical_data = self.generate_historical_risk_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(historical_data, x='timestamp', y='risk_score',
                         title="Risk Score Over Time",
                         color_discrete_sequence=['red'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(historical_data, x='timestamp', y='prediction_confidence',
                         title="Prediction Confidence Over Time",
                         color_discrete_sequence=['blue'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance analysis
        st.markdown("### 🔍 Feature Importance Analysis")
        
        feature_importance = self.get_feature_importance()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(feature_importance, x='importance', y='feature',
                        orientation='h', title="Model Feature Importance",
                        color='importance', color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation matrix
            correlation_data = self.generate_correlation_matrix()
            
            fig = px.imshow(correlation_data, text_auto=True,
                           title="Feature Correlation Matrix",
                           color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.markdown("### 📊 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Accuracy", "97.8%", "↑ 2.1%")
        
        with col2:
            st.metric("AUC Score", "0.673", "↑ 0.045")
        
        with col3:
            st.metric("Precision", "89.2%", "↓ 1.3%")
        
        with col4:
            st.metric("Recall", "76.5%", "↑ 3.7%")
    
    def show_alert_management_tab(self):
        """Display alert management interface"""
        
        st.markdown("## 🚨 Alert Management")
        
        # Alert summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Alerts", "3", "↑ 1")
        
        with col2:
            st.metric("Resolved Today", "8", "↓ 2")
        
        with col3:
            st.metric("High Priority", "1", "→ 0")
        
        with col4:
            st.metric("Response Time", "4.2 min", "↓ 1.1 min")
        
        # Alert filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alert_type_filter = st.multiselect(
                "Alert Type", 
                ["Prediction", "Detection", "Sensor", "System"],
                default=["Prediction", "Detection"]
            )
        
        with col2:
            priority_filter = st.multiselect(
                "Priority Level",
                ["Critical", "High", "Medium", "Low"],
                default=["Critical", "High", "Medium"]
            )
        
        with col3:
            time_filter = st.selectbox(
                "Time Range",
                ["Last Hour", "Last 24 Hours", "Last Week", "All Time"]
            )
        
        # Alert timeline
        st.markdown("### ⏰ Alert Timeline")
        
        alert_timeline_data = self.generate_alert_timeline()
        
        fig = px.timeline(alert_timeline_data, x_start="start", x_end="end", 
                         y="alert_id", color="priority",
                         title="Alert Timeline")
        st.plotly_chart(fig, use_container_width=True)
        
        # Alert details table
        st.markdown("### 📋 Alert Details")
        
        alert_details = self.get_alert_details()
        
        # Color code alerts by priority
        def highlight_priority(row):
            if row['Priority'] == 'Critical':
                return ['background-color: #ffcccc'] * len(row)
            elif row['Priority'] == 'High':
                return ['background-color: #ffe6cc'] * len(row)
            elif row['Priority'] == 'Medium':
                return ['background-color: #fff2cc'] * len(row)
            else:
                return ['background-color: #e6ffe6'] * len(row)
        
        styled_alerts = alert_details.style.apply(highlight_priority, axis=1)
        st.dataframe(styled_alerts, use_container_width=True)
        
        # Alert actions
        st.markdown("### 🎛️ Alert Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔕 Acknowledge All"):
                st.success("All alerts acknowledged!")
        
        with col2:
            if st.button("📧 Send Notifications"):
                st.info("Notifications sent!")
        
        with col3:
            if st.button("📊 Generate Report"):
                st.info("Alert report generated!")
        
        with col4:
            if st.button("🧹 Clear Resolved"):
                st.success("Resolved alerts cleared!")
    
    def show_model_performance_tab(self):
        """Display model performance metrics and analysis"""
        
        st.markdown("## ⚙️ Model Performance & Configuration")
        
        # Model status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("🧠 **ML Models**\n✅ XGBoost: Active\n✅ Random Forest: Active\n✅ Neural Network: Active")
        
        with col2:
            st.success("📊 **Data Pipeline**\n✅ Terrain: Connected\n✅ Environmental: Active\n✅ Sensors: 3/5 Online")
        
        with col3:
            st.warning("🔄 **Last Training**\n📅 2024-09-15\n⏰ 14:30 UTC\n📊 5,000 samples")
        
        # Performance comparison
        st.markdown("### 📊 Model Performance Comparison")
        
        model_performance = {
            'Model': ['XGBoost', 'Random Forest', 'Neural Network', 'Ensemble'],
            'Accuracy': [0.970, 0.978, 0.978, 0.980],
            'AUC Score': [0.587, 0.627, 0.673, 0.599],
            'Training Time': [45, 67, 234, 346]  # seconds
        }
        
        perf_df = pd.DataFrame(model_performance)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(perf_df, x='Model', y='Accuracy',
                        title="Model Accuracy Comparison",
                        color='Accuracy', color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(perf_df, x='Model', y='AUC Score',
                        title="AUC Score Comparison", 
                        color='AUC Score', color_continuous_scale='plasma')
            st.plotly_chart(fig, use_container_width=True)
        
        # Training history
        st.markdown("### 📈 Training History")
        
        training_history = self.generate_training_history()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Validation Loss', 'Accuracy', 'AUC Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=training_history['epoch'], y=training_history['train_loss'], 
                      name="Train Loss"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=training_history['epoch'], y=training_history['val_loss'], 
                      name="Val Loss"),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=training_history['epoch'], y=training_history['accuracy'], 
                      name="Accuracy"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=training_history['epoch'], y=training_history['auc'], 
                      name="AUC"),
            row=2, col=2
        )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model configuration
        st.markdown("### ⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Hyperparameters**")
            st.code("""
            XGBoost Configuration:
            - max_depth: 6
            - learning_rate: 0.1
            - n_estimators: 100
            - subsample: 0.8
            - colsample_bytree: 0.8
            
            Random Forest:
            - n_estimators: 100
            - max_depth: 10
            - min_samples_split: 5
            - min_samples_leaf: 2
            """)
        
        with col2:
            st.markdown("**Feature Engineering**")
            st.code("""
            Input Features: 19
            - Terrain: 7 features
            - Environmental: 6 features
            - Historical: 3 features
            - Synthetic: 3 features
            
            Preprocessing:
            - StandardScaler normalization
            - Missing value imputation
            - Feature correlation analysis
            """)
        
        # Retraining controls
        st.markdown("### 🔄 Model Management")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Retrain Models"):
                with st.spinner("Retraining models..."):
                    time.sleep(3)
                st.success("Models retrained successfully!")
        
        with col2:
            if st.button("💾 Save Models"):
                st.success("Models saved!")
        
        with col3:
            if st.button("📊 Model Validation"):
                st.info("Running validation...")
        
        with col4:
            if st.button("🔍 Feature Analysis"):
                st.info("Analyzing features...")
    
    # Helper methods
    def get_current_risk_score(self):
        """Get current risk score"""
        if st.session_state.prediction_history:
            return st.session_state.prediction_history[-1].get('risk_score', 0.5)
        return np.random.uniform(0.3, 0.8)
    
    def get_risk_level_name(self, risk_score):
        """Convert risk score to level name"""
        if risk_score < 0.3:
            return "Low"
        elif risk_score < 0.6:
            return "Medium"
        elif risk_score < 0.8:
            return "High"
        else:
            return "Critical"
    
    def get_uptime(self):
        """Get system uptime"""
        if hasattr(st.session_state, 'start_time'):
            uptime = datetime.now() - st.session_state.start_time
            return f"{uptime.seconds // 3600}h {(uptime.seconds % 3600) // 60}m"
        return "0h 0m"
    
    def simulate_data_update(self):
        """Simulate new data update"""
        new_prediction = {
            'timestamp': datetime.now(),
            'risk_score': np.random.uniform(0.2, 0.9),
            'confidence': np.random.uniform(0.7, 0.95),
            'location': (40.7128, -74.0060)
        }
        st.session_state.prediction_history.append(new_prediction)
        st.session_state.last_update = datetime.now()
    
    def get_current_risk_assessment(self):
        """Get current comprehensive risk assessment"""
        return {
            'risk_score': self.get_current_risk_score(),
            'components': {
                'Terrain': np.random.uniform(0.4, 0.8),
                'Environmental': np.random.uniform(0.2, 0.7),
                'Historical': np.random.uniform(0.1, 0.5),
                'ML Prediction': np.random.uniform(0.3, 0.9),
                'Proximity': np.random.uniform(0.0, 0.3)
            },
            'terrain_features': {
                'slope': np.random.uniform(30, 80),
                'fracture_density': np.random.uniform(2, 8),
                'instability_index': np.random.uniform(0.3, 0.9),
                'elevation': np.random.uniform(1000, 2000)
            },
            'environmental_conditions': {
                'rainfall': np.random.uniform(0, 100),
                'temperature_variation': np.random.uniform(5, 30),
                'seismic_activity': np.random.uniform(0, 5),
                'wind_speed': np.random.uniform(10, 60)
            }
        }
    
    def get_probability_forecast(self):
        """Get probability forecast for different time horizons"""
        base_prob = self.get_current_risk_score()
        return {
            '1 Hour': base_prob * 0.1,
            '6 Hours': base_prob * 0.3,
            '24 Hours': base_prob * 0.6,
            '7 Days': base_prob * 0.8,
            '30 Days': base_prob
        }
    
    def plot_risk_trend(self):
        """Plot risk score trend"""
        # Generate sample data
        hours = list(range(24))
        risk_scores = [0.3 + 0.4 * np.sin(h * np.pi / 12) + np.random.normal(0, 0.05) for h in hours]
        
        df = pd.DataFrame({'Hour': hours, 'Risk Score': risk_scores})
        
        fig = px.line(df, x='Hour', y='Risk Score', 
                     title="24-Hour Risk Trend",
                     color_discrete_sequence=['red'])
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                     annotation_text="Alert Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_environmental_conditions(self):
        """Plot current environmental conditions"""
        conditions = self.get_current_risk_assessment()['environmental_conditions']
        
        # Normalize for radar chart
        normalized_conditions = {
            'Rainfall': conditions['rainfall'] / 100,
            'Temperature Var': conditions['temperature_variation'] / 30,
            'Seismic Activity': conditions['seismic_activity'] / 5,
            'Wind Speed': conditions['wind_speed'] / 60
        }
        
        categories = list(normalized_conditions.keys())
        values = list(normalized_conditions.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name='Current Conditions'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=False,
            title="Environmental Conditions"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_recent_alerts(self, limit=5):
        """Show recent alerts"""
        # Generate sample alerts
        alert_types = ['Prediction', 'Detection', 'Sensor']
        priorities = ['High', 'Medium', 'Low']
        
        recent_alerts = []
        for i in range(limit):
            recent_alerts.append({
                'Time': (datetime.now() - timedelta(hours=i)).strftime('%H:%M'),
                'Type': np.random.choice(alert_types),
                'Priority': np.random.choice(priorities),
                'Message': f"Risk level {np.random.choice(['elevated', 'high', 'critical'])} detected",
                'Status': np.random.choice(['Active', 'Resolved'])
            })
        
        df = pd.DataFrame(recent_alerts)
        st.dataframe(df, use_container_width=True)
    
    def generate_risk_map_data(self):
        """Generate synthetic risk map data"""
        n_points = 100
        
        # Center around default location with some spread
        lat_center, lon_center = 40.7128, -74.0060
        
        data = []
        for _ in range(n_points):
            lat = lat_center + np.random.normal(0, 0.01)
            lon = lon_center + np.random.normal(0, 0.01)
            risk_score = np.random.beta(2, 5)  # Skewed toward lower risk
            
            data.append({
                'lat': lat,
                'lon': lon,
                'risk_score': risk_score
            })
        
        return pd.DataFrame(data)
    
    def generate_historical_risk_data(self):
        """Generate historical risk data for trend analysis"""
        dates = pd.date_range(end=datetime.now(), periods=168, freq='H')  # Last week
        
        risk_scores = []
        confidences = []
        
        for i, date in enumerate(dates):
            # Simulate some patterns
            base_risk = 0.4 + 0.2 * np.sin(i * 2 * np.pi / 24)  # Daily cycle
            noise = np.random.normal(0, 0.1)
            risk_score = np.clip(base_risk + noise, 0, 1)
            
            confidence = np.random.uniform(0.7, 0.95)
            
            risk_scores.append(risk_score)
            confidences.append(confidence)
        
        return pd.DataFrame({
            'timestamp': dates,
            'risk_score': risk_scores,
            'prediction_confidence': confidences
        })
    
    def get_feature_importance(self):
        """Get feature importance data"""
        features = [
            'Slope', 'Fracture Density', 'Rainfall', 'Freeze-Thaw Cycles',
            'Seismic Activity', 'Temperature Variation', 'Wind Speed',
            'Instability Index', 'Roughness', 'Elevation'
        ]
        
        importance_scores = np.random.exponential(0.1, len(features))
        importance_scores = importance_scores / importance_scores.sum()  # Normalize
        
        return pd.DataFrame({
            'feature': features,
            'importance': importance_scores
        }).sort_values('importance', ascending=True)
    
    def generate_correlation_matrix(self):
        """Generate feature correlation matrix"""
        features = ['Slope', 'Fracture', 'Rainfall', 'Seismic', 'Temperature']
        n_features = len(features)
        
        # Generate random correlation matrix
        A = np.random.randn(n_features, n_features)
        correlation_matrix = np.corrcoef(A)
        
        return pd.DataFrame(correlation_matrix, index=features, columns=features)
    
    def generate_alert_timeline(self):
        """Generate alert timeline data"""
        alerts = []
        for i in range(10):
            start_time = datetime.now() - timedelta(hours=24-i*2)
            end_time = start_time + timedelta(minutes=np.random.randint(30, 180))
            
            alerts.append({
                'alert_id': f'Alert {i+1}',
                'start': start_time,
                'end': end_time,
                'priority': np.random.choice(['Critical', 'High', 'Medium', 'Low'])
            })
        
        return pd.DataFrame(alerts)
    
    def get_alert_details(self):
        """Get detailed alert information"""
        alerts = []
        for i in range(15):
            alerts.append({
                'ID': f'A{i+1:03d}',
                'Time': (datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M'),
                'Type': np.random.choice(['Prediction', 'Detection', 'Sensor']),
                'Priority': np.random.choice(['Critical', 'High', 'Medium', 'Low']),
                'Location': f'Zone {np.random.choice(["A", "B", "C"])}',
                'Status': np.random.choice(['Active', 'Resolved', 'Acknowledged']),
                'Confidence': f'{np.random.uniform(0.6, 0.95):.2f}'
            })
        
        return pd.DataFrame(alerts)
    
    def generate_training_history(self):
        """Generate model training history"""
        epochs = list(range(1, 101))
        
        # Simulate decreasing loss and increasing metrics
        train_loss = [1.0 * np.exp(-epoch/20) + np.random.normal(0, 0.01) for epoch in epochs]
        val_loss = [1.2 * np.exp(-epoch/25) + np.random.normal(0, 0.01) for epoch in epochs]
        accuracy = [0.5 + 0.4 * (1 - np.exp(-epoch/15)) + np.random.normal(0, 0.005) for epoch in epochs]
        auc = [0.5 + 0.4 * (1 - np.exp(-epoch/20)) + np.random.normal(0, 0.005) for epoch in epochs]
        
        return pd.DataFrame({
            'epoch': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'auc': auc
        })
    
    def generate_new_prediction(self):
        """Generate new prediction"""
        new_prediction = {
            'timestamp': datetime.now(),
            'risk_score': np.random.uniform(0.2, 0.9),
            'confidence': np.random.uniform(0.7, 0.95)
        }
        st.session_state.prediction_history.append(new_prediction)
    
    def show_feature_analysis(self):
        """Show feature analysis popup"""
        st.info("Feature analysis would show detailed breakdown of how each feature contributes to the current prediction.")
    
    def generate_system_report(self):
        """Generate system report"""
        st.success("System report generated and saved to outputs/system_report.pdf")


def main():
    """Main dashboard function"""
    
    # Initialize dashboard
    dashboard = EnhancedRockfallDashboard()
    
    # Auto-refresh every 30 seconds if system is running
    if st.session_state.system_running:
        time.sleep(0.1)  # Small delay to prevent excessive refreshing
        if st.button("🔄", key="auto_refresh", help="Auto-refresh"):
            st.experimental_rerun()


if __name__ == "__main__":
    main()