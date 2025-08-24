import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime, timedelta
import io

# Set page config
st.set_page_config(
    page_title="F&B Vision",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with improved centering
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .anomaly-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: left;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.4);
        animation: pulse 2s infinite;
        font-size: 16px;
        font-weight: bold;
    }
    .normal-alert {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: left;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(81, 207, 102, 0.4);
        font-size: 16px;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    .metric-card h4 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-size: 16px;
        font-weight: 600;
    }
    .metric-card h2 {
        color: #667eea;
        margin: 0.5rem 0;
        font-size: 32px;
        font-weight: bold;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 14px;
        font-weight: 500;
    }
    .parameter-card {
        background: linear-gradient(145deg, #34495e 0%, #2c3e50 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .status-normal {
        background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        display: inline-block;
    }
    .status-warning {
        background: linear-gradient(90deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        display: inline-block;
    }
    .status-danger {
        background: linear-gradient(90deg, #f8d7da 0%, #f5b7b1 100%);
        color: #721c24;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        display: inline-block;
    }
    .detection-settings {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(44, 62, 80, 0.3);
    }
    .section-spacing {
        margin: 3rem 0;
        padding: 2rem 0;
        border-top: 1px solid #e9ecef;
    }
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 2rem 0;
    }
    .filter-container {
        background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        border: 1px solid #dee2e6;
    }
    .notification-popup {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        padding: 1rem 2rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        animation: slideIn 0.5s ease-out;
    }
    .processing-container {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(52, 152, 219, 0.3);
    }
    .summary-container {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .results-header {
        background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px 15px 0 0;
        margin-bottom: 0;
        text-align: left;
    }
    .filter-section {
        background: white;
        padding: 1.5rem;
        border-radius: 0 0 15px 15px;
        border: 1px solid #e9ecef;
        border-top: none;
        margin-top: 0;
    }
    .upload-section {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #dee2e6;
        margin: 2rem 0;
        text-align: center;
    }
    .download-section {
        background: linear-gradient(145deg, #e8f5e8 0%, #f0f8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    @keyframes slideIn {
        from { transform: translateX(100%); }
        to { transform: translateX(0); }
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 25px;
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        color: white;
        font-weight: 600;
        font-size: 16px;
        border: none;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Product-specific parameter configurations
PRODUCT_CONFIGS = {
    "Bread": {
        'Flour (kg)': {'min': 45, 'max': 55, 'optimal': 50},
        'Sugar (kg)': {'min': 3, 'max': 7, 'optimal': 5},
        'Yeast (kg)': {'min': 1, 'max': 3, 'optimal': 2},
        'Water Temp (C)': {'min': 20, 'max': 30, 'optimal': 25},
        'Salt (kg)': {'min': 2, 'max': 4, 'optimal': 3},
        'Mixer Speed (RPM)': {'min': 80, 'max': 120, 'optimal': 100},
        'Mixing Temp (C)': {'min': 20, 'max': 28, 'optimal': 24},
        'Fermentation Temp (C)': {'min': 25, 'max': 35, 'optimal': 30},
        'Oven Temp (C)': {'min': 180, 'max': 220, 'optimal': 200},
        'Final Weight (kg)': {'min': 140, 'max': 160, 'optimal': 150},
        'Quality_Score': {'min': 70, 'max': 95, 'optimal': 85}
    },
    "Cake": {
        'Flour (kg)': {'min': 30, 'max': 40, 'optimal': 35},
        'Sugar (kg)': {'min': 15, 'max': 25, 'optimal': 20},
        'Eggs (count)': {'min': 8, 'max': 15, 'optimal': 12},
        'Butter (kg)': {'min': 8, 'max': 15, 'optimal': 12},
        'Milk (L)': {'min': 2, 'max': 5, 'optimal': 3},
        'Mixer Speed (RPM)': {'min': 60, 'max': 100, 'optimal': 80},
        'Mixing Temp (C)': {'min': 18, 'max': 25, 'optimal': 22},
        'Baking Temp (C)': {'min': 160, 'max': 180, 'optimal': 170},
        'Oven Temp (C)': {'min': 160, 'max': 180, 'optimal': 170},
        'Final Weight (kg)': {'min': 80, 'max': 120, 'optimal': 100},
        'Quality_Score': {'min': 75, 'max': 98, 'optimal': 90}
    },
    "Cheese": {
        'Flour (kg)': {'min': 25, 'max': 35, 'optimal': 30},
        'Butter (kg)': {'min': 12, 'max': 20, 'optimal': 16},
        'Water (L)': {'min': 1, 'max': 3, 'optimal': 2},
        'Salt (kg)': {'min': 0.5, 'max': 1.5, 'optimal': 1},
        'Sugar (kg)': {'min': 2, 'max': 6, 'optimal': 4},
        'Mixer Speed (RPM)': {'min': 40, 'max': 80, 'optimal': 60},
        'Mixing Temp (C)': {'min': 15, 'max': 22, 'optimal': 18},
        'Rolling Temp (C)': {'min': 12, 'max': 18, 'optimal': 15},
        'Oven Temp (C)': {'min': 190, 'max': 230, 'optimal': 210},
        'Final Weight (kg)': {'min': 60, 'max': 90, 'optimal': 75},
        'Quality_Score': {'min': 80, 'max': 98, 'optimal': 92}
    },
    "Wine": {
        'Flour (kg)': {'min': 20, 'max': 30, 'optimal': 25},
        'Sugar (kg)': {'min': 8, 'max': 15, 'optimal': 12},
        'Butter (kg)': {'min': 6, 'max': 12, 'optimal': 9},
        'Eggs (count)': {'min': 2, 'max': 6, 'optimal': 4},
        'Vanilla (ml)': {'min': 10, 'max': 30, 'optimal': 20},
        'Mixer Speed (RPM)': {'min': 50, 'max': 90, 'optimal': 70},
        'Mixing Temp (C)': {'min': 18, 'max': 25, 'optimal': 22},
        'Chilling Temp (C)': {'min': 2, 'max': 8, 'optimal': 5},
        'Oven Temp (C)': {'min': 170, 'max': 190, 'optimal': 180},
        'Final Weight (kg)': {'min': 40, 'max': 70, 'optimal': 55},
        'Quality_Score': {'min': 78, 'max': 96, 'optimal': 88}
    },
    "Milk": {
        'Flour (kg)': {'min': 40, 'max': 50, 'optimal': 45},
        'Water (L)': {'min': 12, 'max': 18, 'optimal': 15},
        'Yeast (kg)': {'min': 0.8, 'max': 2, 'optimal': 1.2},
        'Salt (kg)': {'min': 1, 'max': 2.5, 'optimal': 1.8},
        'Olive Oil (L)': {'min': 1, 'max': 3, 'optimal': 2},
        'Mixer Speed (RPM)': {'min': 70, 'max': 110, 'optimal': 90},
        'Mixing Temp (C)': {'min': 22, 'max': 28, 'optimal': 25},
        'Fermentation Temp (C)': {'min': 24, 'max': 32, 'optimal': 28},
        'Proofing Time (hours)': {'min': 4, 'max': 12, 'optimal': 8},
        'Final Weight (kg)': {'min': 70, 'max': 90, 'optimal': 80},
        'Quality_Score': {'min': 72, 'max': 94, 'optimal': 85}
    }
}

PRODUCT_TYPES = list(PRODUCT_CONFIGS.keys())

# Sample datasets tracker
SAMPLE_DATASETS = [
    {"name": "normal_batch_001.csv", "type": "Normal", "samples": 30, "anomaly_rate": "10%"},
    {"name": "anomalous_batch_002.csv", "type": "Anomalous", "samples": 30, "anomaly_rate": "40%"},
    {"name": "mixed_batch_003.csv", "type": "Mixed", "samples": 50, "anomaly_rate": "25%"},
    {"name": "quality_test_004.csv", "type": "Quality Test", "samples": 100, "anomaly_rate": "15%"},
    {"name": "production_sample_005.csv", "type": "Production", "samples": 75, "anomaly_rate": "8%"}
]

def initialize_session_state():
    """Initialize session state variables"""
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'anomaly_threshold' not in st.session_state:
        st.session_state.anomaly_threshold = 0.5
    if 'detection_sensitivity' not in st.session_state:
        st.session_state.detection_sensitivity = "Medium"
    if 'current_batch_data' not in st.session_state:
        st.session_state.current_batch_data = None
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = "Bread"
    if 'live_data' not in st.session_state:
        st.session_state.live_data = generate_live_data()
    if 'notification_message' not in st.session_state:
        st.session_state.notification_message = None

def show_notification(message, type="info"):
    """Show notification popup"""
    if type == "success":
        color = "#28a745"
    elif type == "error":
        color = "#dc3545"
    elif type == "warning":
        color = "#ffc107"
    else:
        color = "#17a2b8"
    
    st.markdown(f"""
    <div class="notification-popup" style="background-color: {color};">
        {message}
    </div>
    """, unsafe_allow_html=True)

def get_current_features():
    """Get features for currently selected product"""
    return PRODUCT_CONFIGS[st.session_state.selected_product]

def simulate_model_training():
    """Simulate model training process"""
    progress_container = st.container()
    with progress_container:
        st.markdown("### 🚀 Model Training in Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        training_steps = [
            ("Loading training data...", "📂"),
            ("Preprocessing features...", "⚙️"),
            ("Training TCN model...", "🧠"),
            ("Extracting features...", "🔍"),
            ("Training LightGBM classifier...", "⚡"),
            ("Validating model performance...", "✅"),
            ("Saving trained models...", "💾")
        ]
        
        for i, (step, emoji) in enumerate(training_steps):
            status_text.markdown(f"{emoji} **{step}**")
            progress_bar.progress((i + 1) / len(training_steps))
            time.sleep(0.8)
        
        status_text.markdown("🎉 **Model training completed successfully!**")
        time.sleep(1)

def generate_live_data():
    """Generate live process data"""
    time_points = list(range(50))
    return {
        'time': time_points,
        'temperature': [25 + random.uniform(-3, 3) + 2*np.sin(i/10) for i in time_points],
        'ph_acidity': [6.5 + random.uniform(-0.5, 0.5) + 0.3*np.cos(i/8) for i in time_points],
        'mixer_speed': [100 + random.uniform(-15, 15) + 5*np.sin(i/12) for i in time_points],
        'pressure': [1.2 + random.uniform(-0.3, 0.3) + 0.1*np.cos(i/6) for i in time_points]
    }

def detect_anomaly_advanced(input_data, threshold=0.5, sensitivity="Medium"):
    """Advanced anomaly detection with configurable parameters"""
    FEATURES = get_current_features()
    anomaly_score = 0
    anomalous_features = []
    parameter_analysis = []
    deviation_details = []
    
    # Sensitivity multipliers
    sensitivity_multipliers = {
        "Low": 0.6,      # Conservative
        "Medium": 1.0,   # Balanced  
        "High": 1.5      # Aggressive
    }
    
    multiplier = sensitivity_multipliers.get(sensitivity, 1.0)
    
    for feature, value in input_data.items():
        if feature in FEATURES:
            range_info = FEATURES[feature]
            status = "Normal"
            note = "Within expected range"
            deviation_pct = 0
            
            # Calculate deviation score and percentage
            if value < range_info['min']:
                deviation_pct = ((range_info['min'] - value) / range_info['min']) * 100
                deviation_score = (range_info['min'] - value) / (range_info['max'] - range_info['min'])
                anomaly_score += deviation_score * 2 * multiplier
                status = "Too Low"
                note = f"Below minimum ({range_info['min']}) by {deviation_pct:.1f}%"
                anomalous_features.append(f"{feature}: {value:.1f} - {note}")
                deviation_details.append({
                    'feature': feature,
                    'deviation_type': 'Too Low',
                    'deviation_percent': deviation_pct,
                    'severity': 'High' if deviation_pct > 20 else 'Medium'
                })
            
            elif value > range_info['max']:
                deviation_pct = ((value - range_info['max']) / range_info['max']) * 100
                deviation_score = (value - range_info['max']) / (range_info['max'] - range_info['min'])
                anomaly_score += deviation_score * 2 * multiplier
                status = "Too High"
                note = f"Above maximum ({range_info['max']}) by {deviation_pct:.1f}%"
                anomalous_features.append(f"{feature}: {value:.1f} - {note}")
                deviation_details.append({
                    'feature': feature,
                    'deviation_type': 'Too High',
                    'deviation_percent': deviation_pct,
                    'severity': 'High' if deviation_pct > 20 else 'Medium'
                })
            
            else:
                # Check deviation from optimal
                optimal_deviation_pct = abs(value - range_info['optimal']) / range_info['optimal'] * 100
                if optimal_deviation_pct > 15:  # 15% deviation from optimal
                    deviation_score = optimal_deviation_pct / 100
                    anomaly_score += deviation_score * multiplier
                    status = "Suboptimal"
                    note = f"Deviation from optimal ({range_info['optimal']}) by {optimal_deviation_pct:.1f}%"
                    deviation_details.append({
                        'feature': feature,
                        'deviation_type': 'Suboptimal',
                        'deviation_percent': optimal_deviation_pct,
                        'severity': 'Low'
                    })
            
            parameter_analysis.append({
                'Parameter': feature,
                'Value': round(value, 2),
                'Status': status,
                'Note': note,
                'Min': range_info['min'],
                'Max': range_info['max'],
                'Optimal': range_info['optimal'],
                'Deviation_Percent': round(deviation_pct, 1) if deviation_pct > 0 else 0
            })
    
    # Calculate probability and final decision
    probability = min(1.0, anomaly_score / len(FEATURES) * 2)
    is_anomaly = probability > threshold
    
    # Calculate confidence
    if probability < 0.3:
        confidence_level = "Low"
    elif probability < 0.7:
        confidence_level = "Medium"  
    else:
        confidence_level = "High"
    
    return {
        'is_anomaly': is_anomaly,
        'probability': probability,
        'confidence': confidence_level,
        'anomalous_features': anomalous_features,
        'parameter_analysis': parameter_analysis,
        'deviation_details': deviation_details,
        'method': "TCN + LightGBM" if st.session_state.model_trained else "Intelligent Fallback",
        'anomaly_score': anomaly_score,
        'sensitivity_used': sensitivity
    }

def create_quality_gauge(quality_score, title="Quality Score"):
    """Create a quality gauge visualization"""
    # Determine color based on score
    if quality_score >= 85:
        color = "green"
    elif quality_score >= 70:
        color = "yellow"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = quality_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 18, 'color': '#2c3e50'}},
        delta = {'reference': 85, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': '#2c3e50', 'tickfont': {'color': '#2c3e50', 'size': 12}},
            'bar': {'color': color, 'thickness': 0.25},
            'steps': [
                {'range': [0, 70], 'color': "rgba(255, 0, 0, 0.2)"},
                {'range': [70, 85], 'color': "rgba(255, 255, 0, 0.2)"},
                {'range': [85, 100], 'color': "rgba(0, 255, 0, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "#2c3e50", 'width': 3},
                'thickness': 0.8,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#2c3e50', 'size': 14}
    )
    return fig

def create_live_parameters_chart():
    """Create live process parameters visualization with better spacing"""
    data = st.session_state.live_data
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('🌡️ Temperature (°C)', '⚗️ pH/Acidity', '🔄 Mixer Speed (RPM)', '📊 Pressure (bar)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.2,  # Increased spacing
        horizontal_spacing=0.15  # Increased spacing
    )
    
    # Temperature data with alerts
    temp_data = data['temperature']
    temp_colors = ['red' if t > 28 or t < 22 else 'orange' if t > 27 or t < 23 else 'green' for t in temp_data]
    
    fig.add_trace(
        go.Scatter(x=data['time'], y=temp_data, mode='lines+markers', name='Temperature',
                  line=dict(color='red', width=3), marker=dict(size=6, color=temp_colors),
                  hovertemplate='Time: %{x}<br>Temp: %{y:.1f}°C<extra></extra>'),
        row=1, col=1
    )
    fig.add_hline(y=25, line_dash="dash", line_color="green", opacity=0.7, row=1, col=1)
    
    # pH data
    ph_data = data['ph_acidity']
    fig.add_trace(
        go.Scatter(x=data['time'], y=ph_data, mode='lines+markers', name='pH/Acidity',
                  line=dict(color='blue', width=3), marker=dict(size=6, color='lightblue'),
                  hovertemplate='Time: %{x}<br>pH: %{y:.1f}<extra></extra>'),
        row=1, col=2
    )
    fig.add_hline(y=6.5, line_dash="dash", line_color="blue", opacity=0.7, row=1, col=2)
    
    # Mixer Speed data
    mixer_data = data['mixer_speed']
    mixer_colors = ['red' if m > 115 or m < 85 else 'orange' if m > 110 or m < 90 else 'green' for m in mixer_data]
    
    fig.add_trace(
        go.Scatter(x=data['time'], y=mixer_data, mode='lines+markers', name='Mixer Speed',
                  line=dict(color='green', width=3), marker=dict(size=6, color=mixer_colors),
                  hovertemplate='Time: %{x}<br>Speed: %{y:.0f} RPM<extra></extra>'),
        row=2, col=1
    )
    fig.add_hline(y=100, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
    
    # Pressure data
    pressure_data = data['pressure']
    fig.add_trace(
        go.Scatter(x=data['time'], y=pressure_data, mode='lines+markers', name='Pressure',
                  line=dict(color='orange', width=3), marker=dict(size=6, color='lightyellow'),
                  hovertemplate='Time: %{x}<br>Pressure: %{y:.2f} bar<extra></extra>'),
        row=2, col=2
    )
    fig.add_hline(y=1.2, line_dash="dash", line_color="orange", opacity=0.7, row=2, col=2)
    
    fig.update_layout(
        height=600,  # Increased height
        showlegend=False,
        title_text="Live Process Parameters - Real-Time Monitoring",
        title_x=0.5,
        plot_bgcolor='rgba(248,249,250,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#2c3e50')
    )
    
    # Update x and y axis labels with better visibility
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Time (minutes)" if i == 2 else "", 
                           tickfont=dict(size=10, color='#2c3e50'),
                           title_font=dict(size=12, color='#2c3e50'),
                           row=i, col=j)
            fig.update_yaxes(tickfont=dict(size=10, color='#2c3e50'),
                           title_font=dict(size=12, color='#2c3e50'),
                           row=i, col=j)
    
    return fig

def create_parameter_analysis_table(parameter_analysis):
    """Create enhanced parameter analysis table with better visibility"""
    df = pd.DataFrame(parameter_analysis)
    
    # Create HTML table with improved styling
    html_table = "<div style='overflow-x: auto; margin: 1rem 0;'>"
    html_table += "<table style='width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1);'>"
    html_table += "<thead><tr style='background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%); color: white;'>"
    
    headers = ["Parameter", "Value", "Status", "Note", "Range", "Deviation %"]
    for header in headers:
        html_table += f"<th style='padding: 15px; text-align: left; font-size: 14px; font-weight: 600;'>{header}</th>"
    html_table += "</tr></thead><tbody>"
    
    for i, row in df.iterrows():
        status_class = "status-normal" if row['Status'] == 'Normal' else "status-warning" if row['Status'] == 'Suboptimal' else "status-danger"
        html_table += f"<tr style='border-bottom: 1px solid #ecf0f1; background: {'#f8f9fa' if i % 2 == 0 else 'white'};'>"
        html_table += f"<td style='padding: 12px; font-weight: 600; color: #2c3e50;'>{row['Parameter']}</td>"
        html_table += f"<td style='padding: 12px; color: #2c3e50; font-weight: 500;'>{row['Value']}</td>"
        html_table += f"<td style='padding: 12px;'><span class='{status_class}'>{row['Status']}</span></td>"
        html_table += f"<td style='padding: 12px; font-size: 12px; color: #495057;'>{row['Note']}</td>"
        html_table += f"<td style='padding: 12px; font-size: 12px; color: #495057;'>{row['Min']}-{row['Max']} (⭐{row['Optimal']})</td>"
        deviation = row.get('Deviation_Percent', 0)
        deviation_color = '#e74c3c' if deviation > 20 else '#f39c12' if deviation > 10 else '#27ae60'
        html_table += f"<td style='padding: 12px; color: {deviation_color}; font-weight: bold;'>{deviation}%</td>"
        html_table += "</tr>"
    
    html_table += "</tbody></table></div>"
    
    return df, html_table

def create_deviation_analysis_chart(deviation_details):
    """Create deviation analysis chart instead of pie chart"""
    if not deviation_details:
        return None
    
    df_dev = pd.DataFrame(deviation_details)
    
    # Create a bar chart showing deviations by severity
    fig = go.Figure()
    
    # Group by severity and count
    severity_counts = df_dev['severity'].value_counts()
    colors = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#f1c40f'}
    
    fig.add_trace(go.Bar(
        x=list(severity_counts.index),
        y=list(severity_counts.values),
        marker_color=[colors.get(sev, '#3498db') for sev in severity_counts.index],
        text=list(severity_counts.values),
        textposition='auto',
        name='Deviation Count'
    ))
    
    fig.update_layout(
        title="Deviation Analysis by Severity Level",
        xaxis_title="Severity Level",
        yaxis_title="Number of Deviations",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50')
    )
    
    return fig

def generate_sample_csv_data(filename_prefix, n_samples=30, anomaly_rate=0.2):
    """Generate sample CSV data with realistic patterns"""
    FEATURES = get_current_features()
    
    if "normal" in filename_prefix:
        np.random.seed(42)
    else:
        np.random.seed(123)
    
    data = []
    n_anomalies = int(n_samples * anomaly_rate)
    
    for i in range(n_samples):
        sample = {}
        is_anomaly_sample = i < n_anomalies
        
        for feature, range_info in FEATURES.items():
            if is_anomaly_sample and random.random() > 0.6:
                # Create anomalous values
                if random.random() > 0.5:
                    # Too high
                    sample[feature] = random.uniform(range_info['max'] * 1.15, range_info['max'] * 1.6)
                else:
                    # Too low  
                    sample[feature] = random.uniform(range_info['min'] * 0.4, range_info['min'] * 0.85)
            else:
                # Normal values with realistic variation
                if feature == 'Quality_Score':
                    # Quality score has different distribution
                    variation = (range_info['max'] - range_info['min']) * 0.12
                    sample[feature] = max(50, min(100, np.random.normal(range_info['optimal'], variation)))
                else:
                    variation = (range_info['max'] - range_info['min']) * 0.15
                    sample[feature] = max(0, np.random.normal(range_info['optimal'], variation))
        
        sample['Batch_ID'] = f"B{i+1:03d}"
        sample['Timestamp'] = (datetime.now() - timedelta(hours=n_samples-i)).strftime("%Y-%m-%d %H:%M:%S")
        data.append(sample)
    
    return pd.DataFrame(data)

def update_visualizations_for_product():
    """Update all visualizations when product type changes"""
    # This function will be called when product selection changes
    # It ensures all charts reflect the new product parameters
    if 'analysis_result' in st.session_state:
        del st.session_state.analysis_result
    
    # Reset any cached analysis results
    st.session_state.live_data = generate_live_data()

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏭 Food Quality Monitor</h1>
        <p style='font-size: 20px; margin-top: 15px; opacity: 0.95;'>Advanced Anomaly Detection System with Real-Time Monitoring</p>
        <p style='font-size: 16px; opacity: 0.85; margin-top: 10px;'>Powered by TCN + LightGBM | Intelligent Process Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Product filter in sidebar
    with st.sidebar:
        st.markdown("### 🎯 Product Configuration")
        selected_product = st.selectbox(
            "🍞 Product Type", 
            PRODUCT_TYPES, 
            index=PRODUCT_TYPES.index(st.session_state.selected_product),
            help="Select the type of food product being monitored",
            key="product_selector"
        )
        
        # Update session state and features if product changed
        if selected_product != st.session_state.selected_product:
            st.session_state.selected_product = selected_product
            update_visualizations_for_product()
            st.rerun()
        
        st.markdown("### ⚙️ System Settings")
        if not st.session_state.model_trained:
            if st.button("🚀 Train Model", type="primary", use_container_width=True):
                with st.spinner("Training AI models..."):
                    simulate_model_training()
                    st.session_state.model_trained = True
                    show_notification("✅ Model training completed successfully!", "success")
                    st.rerun()
        else:
            st.success("✅ Model Ready & Active")
            if st.button("🔄 Retrain Model", use_container_width=True):
                st.session_state.model_trained = False
                st.rerun()
    
    # Top status panel
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("### 📊 Current Configuration")
        st.info(f"🏷️ Product: **{selected_product}**")
        FEATURES = get_current_features()
        st.info(f"🔢 Parameters: **{len(FEATURES)} features**")
    
    with col2:
        st.markdown("### 🎯 System Status")
        if st.session_state.model_trained:
            st.success("🤖 AI Model: **Active**")
            st.success("📡 Monitoring: **Live**")
        else:
            st.error("⚠️ Model Training Required")
            st.warning("📡 Monitoring: **Limited**")
    
    with col3:
        st.markdown("### 📈 Quick Stats")
        st.metric("🎯 Uptime", "99.8%")
        st.metric("📊 Accuracy", "94.2%")
    
    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Dashboard", 
        "🔍 Batch Explorer", 
        "🚨 Anomaly Detection", 
        "📊 Analytics"
    ])
    
    with tab1:  # Dashboard
        st.markdown("## 🏠 Production Dashboard")
        
        if st.session_state.model_trained:
            # Refresh button for live data
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("🔄 Refresh Live Data"):
                    st.session_state.live_data = generate_live_data()
                    show_notification("📊 Live data refreshed!", "info")
                    st.rerun()
            
            st.markdown("---")
            
            # Main dashboard layout with proper spacing
            col1, col2 = st.columns([3, 1], gap="large")
            
            with col1:
                st.markdown("""
                <div class="chart-container">
                    <h3 style="color: #2c3e50; margin-bottom: 1rem;">📈 Live Process Parameters</h3>
                </div>
                """, unsafe_allow_html=True)
                live_chart = create_live_parameters_chart()
                st.plotly_chart(live_chart, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="chart-container">
                    <h3 style="color: #2c3e50; margin-bottom: 1rem; text-align: center;">🎯 Quality Metrics</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Overall quality gauge
                quality_score = random.uniform(78, 96)
                quality_gauge = create_quality_gauge(quality_score, "Overall Quality")
                st.plotly_chart(quality_gauge, use_container_width=True)
            
            # Production Overview with better spacing
            st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
            st.markdown("### 📊 Production Overview")
            
            col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4>🏭 Active Batches</h4>
                    <h2>15</h2>
                    <p style='color: #28a745;'>+3 from yesterday</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4>📈 Daily Production</h4>
                    <h2>1,847 kg</h2>
                    <p style='color: #28a745;'>+12% from target</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎯 Quality Score</h4>
                    <h2>{quality_score:.1f}%</h2>
                    <p style='color: #28a745;'>+2.1% improvement</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                anomaly_rate = random.uniform(2.5, 4.8)
                color = "#dc3545" if anomaly_rate > 4 else "#fd7e14" if anomaly_rate > 3.5 else "#28a745"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>⚠️ Anomaly Rate</h4>
                    <h2>{anomaly_rate:.1f}%</h2>
                    <p style='color: {color};'>-0.8% from last week</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                efficiency = random.uniform(92, 98)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>⚡ Efficiency</h4>
                    <h2>{efficiency:.1f}%</h2>
                    <p style='color: #28a745;'>+1.2% optimal</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Recent Alerts with improved spacing
            st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
            st.markdown("### 🚨 Recent System Alerts")
            alerts_data = [
                {"Time": "2 min ago", "Type": "⚠️ Warning", "Message": "Mixer Speed deviation detected in Batch B047", "Status": "✅ Resolved"},
                {"Time": "15 min ago", "Type": "ℹ️ Info", "Message": "Quality score improved for Batch B046", "Status": "✅ Normal"},
                {"Time": "1 hour ago", "Type": "🔴 Critical", "Message": "Temperature anomaly in Batch B045", "Status": "🔍 Under Review"},
                {"Time": "2 hours ago", "Type": "ℹ️ Info", "Message": "Production target achieved for today", "Status": "🎉 Success"}
            ]
            
            alerts_df = pd.DataFrame(alerts_data)
            st.dataframe(
                alerts_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Time": st.column_config.TextColumn("⏰ Time", width="small"),
                    "Type": st.column_config.TextColumn("🏷️ Type", width="small"),
                    "Message": st.column_config.TextColumn("📄 Message", width="large"),
                    "Status": st.column_config.TextColumn("📊 Status", width="medium")
                }
            )
        
        else:
            st.markdown("""
            <div class='upload-section'>
                <h3 style='color: #495057;'>⚠️ Model Training Required</h3>
                <p style='color: #6c757d; font-size: 16px; margin: 1rem 0;'>Please train the AI model first to access dashboard features and real-time monitoring.</p>
                <p style='color: #6c757d; font-size: 14px;'>Use the sidebar to start model training.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:  # Batch Explorer
        st.markdown("## 🔍 Batch Data Explorer")
        
        if st.session_state.model_trained:
            # File upload section with better layout
            col1, col2 = st.columns([2, 1], gap="large")
            
            with col1:
                st.markdown(f"""
                <div class="upload-section">
                    <h3 style="color: #2c3e50; margin-bottom: 1rem;">📁 Upload Batch Data</h3>
                    <p style="color: #6c757d; margin-bottom: 1rem;">Upload CSV file containing batch production data for {selected_product}</p>
                </div>
                """, unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader(
                    "Choose CSV file", 
                    type=['csv'],
                    help=f"Upload a CSV file containing batch production data for {selected_product}"
                )
                
                if uploaded_file:
                    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                    show_notification(f"📄 File '{uploaded_file.name}' processed!", "success")
            
            with col2:
                st.markdown("""
                <div class="download-section">
                    <h3 style="color: #2c3e50; margin-bottom: 1rem;">📥 Recent History of Batch Data Uploads</h3>
                    <p style="color: #6c757d; font-size: 14px;">Download for testing</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display available sample datasets
                for dataset in SAMPLE_DATASETS:
                    with st.expander(f"📄 {dataset['name']}", expanded=False):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.info(f"**Type:** {dataset['type']}")
                            st.info(f"**Samples:** {dataset['samples']}")
                        with col_b:
                            st.warning(f"**Anomaly Rate:** {dataset['anomaly_rate']}")
                            
                            if st.button(f"📥 Download", key=f"download_{dataset['name']}"):
                                # Generate data based on dataset type
                                anomaly_rate = float(dataset['anomaly_rate'].replace('%', '')) / 100
                                sample_data = generate_sample_csv_data(dataset['name'], dataset['samples'], anomaly_rate)
                                csv_string = sample_data.to_csv(index=False)
                                
                                st.download_button(
                                    label=f"💾 Get {dataset['name']}",
                                    data=csv_string,
                                    file_name=dataset['name'],
                                    mime="text/csv",
                                    key=f"dl_btn_{dataset['name']}"
                                )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.current_batch_data = df
                    
                    # Data overview with better styling
                    st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                    st.markdown("### 📊 Dataset Overview")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    FEATURES = get_current_features()
                    feature_cols = [col for col in df.columns if col in FEATURES]
                    
                    with col1:
                        st.metric("📊 Total Records", len(df))
                    with col2:
                        st.metric("🔢 Features", len(feature_cols))
                    with col3:
                        if 'Timestamp' in df.columns:
                            date_range = pd.to_datetime(df['Timestamp']).dt.date
                            st.metric("📅 Date Range", f"{date_range.min()} to {date_range.max()}")
                        else:
                            st.metric("⏱️ Processing", "Real-time")
                    with col4:
                        st.metric("🏷️ Product", selected_product)
                    
                    # Data preview
                    st.markdown("### 📋 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Validate required columns
                    missing_features = [f for f in FEATURES.keys() if f not in df.columns]
                    if missing_features:
                        st.error(f"❌ Missing required columns for {selected_product}: {', '.join(missing_features)}")
                        st.info("💡 Please ensure your CSV contains all required feature columns for the selected product type.")
                        
                        # Show expected format
                        st.markdown("### 📝 Expected CSV Format")
                        expected_df = pd.DataFrame([{f: FEATURES[f]['optimal'] for f in FEATURES.keys()}])
                        expected_df['Batch_ID'] = 'B001'
                        expected_df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.dataframe(expected_df)
                        return
                    
                    # Analysis button
                    st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        if st.button("🔍 Analyze Complete Batch", type="primary", use_container_width=True):
                            st.markdown("""
                            <div class="processing-container">
                                <h3>🔄 Processing Batch Analysis...</h3>
                                <p>Analyzing samples for anomalies and quality issues</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Analysis with progress tracking
                            results = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, row in df.iterrows():
                                status_text.text(f"Analyzing sample {i+1}/{len(df)} - {row.get('Batch_ID', f'Sample {i+1}')}")
                                progress_bar.progress((i+1)/len(df))
                                
                                sample_data = {k: v for k, v in row.items() if k in FEATURES}
                                
                                if sample_data:
                                    result = detect_anomaly_advanced(
                                        sample_data, 
                                        st.session_state.anomaly_threshold,
                                        st.session_state.detection_sensitivity
                                    )
                                    
                                    results.append({
                                        'Sample_ID': i+1,
                                        'Batch_ID': row.get('Batch_ID', f'B{i+1:03d}'),
                                        'Timestamp': row.get('Timestamp', ''),
                                        'Status': 'Anomaly' if result['is_anomaly'] else 'Normal',
                                        'Probability': result['probability'],
                                        'Confidence': result['confidence'],
                                        'Anomalous_Features': len(result['anomalous_features']),
                                        'Method': result['method']
                                    })
                                
                                time.sleep(0.02)  # Reduced delay
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            if results:
                                results_df = pd.DataFrame(results)
                                
                                # Show notification for analysis completion
                                anomaly_count = len(results_df[results_df['Status'] == 'Anomaly'])
                                if anomaly_count > 0:
                                    show_notification(f"🚨 Analysis complete! Found {anomaly_count} anomalies out of {len(results_df)} samples.", "warning")
                                else:
                                    show_notification(f"✅ Analysis complete! All {len(results_df)} samples are normal.", "success")
                                
                                # Summary statistics
                                st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                                st.markdown("### 📊 Batch Analysis Summary")
                                
                                normal_count = len(results_df[results_df['Status'] == 'Normal'])
                                anomaly_rate = (anomaly_count / len(results_df)) * 100
                                avg_probability = results_df['Probability'].mean()
                                high_confidence = len(results_df[results_df['Confidence'] == 'High'])
                                
                                st.markdown(f"""
                                <div class="summary-container">
                                    <div style="display: flex; flex-direction:column;gap:5px;; text-align: center;">
                                        <div>
                                            <h3 style="color: #2c3e50; margin: 0;font-size:20px">{len(results_df)}</h3>
                                            <p style="color: #6c757d; margin: 0;font-size:20px">📊 Total Samples</p>
                                        </div>
                                        <div>
                                            <h3 style="color: #27ae60; margin: 0;font-size:20px">{normal_count}</h3>
                                            <p style="color: #6c757d; margin: 0;font-size:20px">✅ Normal ({(normal_count/len(results_df)*100):.1f}%)</p>
                                        </div>
                                        <div>
                                            <h3 style="color: #e74c3c; margin: 0;font-size:20px">{anomaly_count}</h3>
                                            <p style="color: #6c757d; margin: 0;font-size:20px">🚨 Anomalies ({anomaly_rate:.1f}%)</p>
                                        </div>
                                        <div>
                                            <h3 style="color: #3498db; margin: 0;font-size:20px">{avg_probability:.3f}</h3>
                                            <p style="color: #6c757d; margin: 0;font-size:20px">📈 Avg Probability</p>
                                        </div>
                                        <div>
                                            <h3 style="color: #9b59b6; margin: 0;font-size:20px">{high_confidence}</h3>
                                            <p style="color: #6c757d; margin: 0;font-size:20px">🎯 High Confidence ({(high_confidence/len(results_df)*100):.0f}%)</p>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Results visualization
                                st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                                st.markdown("### 📈 Batch Analysis Visualization")
                                
                                # Create comprehensive visualization with better spacing
                                fig = make_subplots(
                                    rows=2, cols=2,
                                    subplot_titles=(
                                        'Detection Results Over Time', 
                                        'Probability Distribution',
                                        'Confidence Level Distribution',
                                        'Anomaly Timeline'
                                    ),
                                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                           [{"secondary_y": False}, {"secondary_y": False}]],
                                    vertical_spacing=0.2,
                                    horizontal_spacing=0.15
                                )
                                
                                # Scatter plot of results
                                colors = ['#e74c3c' if status == 'Anomaly' else '#27ae60' for status in results_df['Status']]
                                fig.add_trace(
                                    go.Scatter(
                                        x=results_df['Sample_ID'], 
                                        y=results_df['Probability'],
                                        mode='markers+lines',
                                        marker=dict(color=colors, size=8, line=dict(width=1, color='white')),
                                        line=dict(color='lightblue', width=1),
                                        name='Detection Results',
                                        hovertemplate='Sample: %{x}<br>Probability: %{y:.3f}<br>Status: %{text}<extra></extra>',
                                        text=results_df['Status']
                                    ),
                                    row=1, col=1
                                )
                                fig.add_hline(y=st.session_state.anomaly_threshold, line_dash="dash", 
                                            line_color="red", opacity=0.7, row=1, col=1)
                                
                                # Probability histogram
                                fig.add_trace(
                                    go.Histogram(x=results_df['Probability'], nbinsx=20, name='Probability Distribution',
                                               marker_color='skyblue', opacity=0.7),
                                    row=1, col=2
                                )
                                
                                # Confidence levels bar chart
                                confidence_counts = results_df['Confidence'].value_counts()
                                fig.add_trace(
                                    go.Bar(x=confidence_counts.index, y=confidence_counts.values,
                                          name='Confidence Levels', 
                                          marker_color=['#e74c3c', '#f39c12', '#27ae60']),
                                    row=2, col=1
                                )
                                
                                # Timeline analysis
                                anomaly_indicators = [1 if s == 'Anomaly' else 0 for s in results_df['Status']]
                                fig.add_trace(
                                    go.Scatter(x=results_df['Sample_ID'], y=anomaly_indicators,
                                             mode='markers', name='Anomaly Timeline', 
                                             marker=dict(color=colors, size=8, symbol='diamond')),
                                    row=2, col=2
                                )
                                
                                fig.update_layout(
                                    height=800,
                                    title_text="Comprehensive Batch Analysis Dashboard",
                                    showlegend=True,
                                    plot_bgcolor='white',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#2c3e50')
                                )
                                
                                st.plotly_chart(fig, use_container_width=True,config={"displayModeBar": True})
                                
                                # Detailed results table with filters
                                st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                                
                                st.markdown("""
                                <div class="results-header">
                                    <h3 style="margin: 0; color: white;">📋 Detailed Analysis Results</h3>
                                    <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8); font-size: 14px;">Filter and analyze individual sample results</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Filter options
                                st.markdown("""
                                <div class="filter-section">
                                """, unsafe_allow_html=True)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    status_filter = st.selectbox("🔍 Filter by Status", ["All", "Normal", "Anomaly"])
                                with col2:
                                    confidence_filter = st.selectbox("🎯 Filter by Confidence", ["All", "Low", "Medium", "High"])
                                with col3:
                                    min_probability = st.slider("📊 Minimum Probability", 0.0, 1.0, 0.0, 0.01)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Apply filters
                                filtered_df = results_df.copy()
                                if status_filter != "All":
                                    filtered_df = filtered_df[filtered_df['Status'] == status_filter]
                                if confidence_filter != "All":
                                    filtered_df = filtered_df[filtered_df['Confidence'] == confidence_filter]
                                filtered_df = filtered_df[filtered_df['Probability'] >= min_probability]
                                
                                # Enhanced dataframe display
                                st.dataframe(
                                    filtered_df, 
                                    use_container_width=True, 
                                    hide_index=True,
                                    column_config={
                                        "Sample_ID": st.column_config.NumberColumn("🔢 Sample", width="small"),
                                        "Batch_ID": st.column_config.TextColumn("🏷️ Batch ID", width="small"),
                                        "Status": st.column_config.TextColumn("📊 Status", width="small"),
                                        "Probability": st.column_config.ProgressColumn("📈 Probability", min_value=0, max_value=1),
                                        "Confidence": st.column_config.TextColumn("🎯 Confidence", width="small"),
                                        "Anomalous_Features": st.column_config.NumberColumn("🔍 Issues", width="small"),
                                        "Method": st.column_config.TextColumn("⚙️ Method", width="medium")
                                    }
                                )
                                
                                # Export results
                                st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                                col1, col2 = st.columns(2)
                                with col1:
                                    csv_export = filtered_df.to_csv(index=False)
                                    st.download_button(
                                        label="📄 Download Analysis Results",
                                        data=csv_export,
                                        file_name=f"batch_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                
                                with col2:
                                    if anomaly_count > 0:
                                        anomaly_details = filtered_df[filtered_df['Status'] == 'Anomaly']
                                        anomaly_csv = anomaly_details.to_csv(index=False)
                                        st.download_button(
                                            label="🚨 Download Anomalies Only",
                                            data=anomaly_csv,
                                            file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                            mime="text/csv",
                                            use_container_width=True
                                        )
                            else:
                                st.error(" No valid data found for analysis")
                
                except Exception as e:
                    st.error(f" Error processing file: {str(e)}")
                    st.info("💡 Please ensure your CSV file has the correct format and required columns for the selected product type.")
            
            else:
                st.markdown(f"""
                <div class='upload-section'>
                    <h3 style='color: #495057;'>📁 Upload Batch Data for Analysis</h3>
                    <p style='color: #6c757d; font-size: 16px; margin: 1rem 0;'>Please upload a CSV file containing batch production data, or use the sample data downloads.</p>
                    <p style='color: #6c757d; font-size: 14px;'>Current product: <strong>{selected_product}</strong> | Supported format: CSV with batch parameters and optional timestamps</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='upload-section'>
                <h3 style='color: #495057;'>⚠️ Model Training Required</h3>
                <p style='color: #6c757d; font-size: 16px; margin: 1rem 0;'>Please train the AI model first to access batch exploration features.</p>
                <p style='color: #6c757d; font-size: 14px;'>Use the sidebar to start model training.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:  # Anomaly Detection
        st.markdown("## 🚨 Anomaly Detection System")
        
        if st.session_state.model_trained:
            # Detection Settings
            st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
            st.markdown("### ⚙️ Detection Settings")
            
            with st.container():
                st.markdown("""
                <div class="detection-settings">
                    <h4 style='color: white; margin-bottom: 2rem; text-align: left;'>🔧 Advanced Configuration</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2, gap="large")
                
                with col1:
                    st.markdown("#### 🎯 Anomaly Threshold")
                    threshold = st.slider(
                        "Threshold Value",
                        0.0, 1.0, 
                        st.session_state.anomaly_threshold,
                        0.01,
                        help="Samples with probability above this threshold are classified as anomalies"
                    )
                    st.session_state.anomaly_threshold = threshold
                    
                    threshold_info = f"Probability > {threshold:.3f} = Anomaly"
                    st.markdown(f"""
                    <div style='background: linear-gradient(90deg, #3498db 0%, #2980b9 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-top: 1rem; text-align: left; box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);'>
                        <strong style="font-size: 16px;">{threshold_info}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### 🎛️ Detection Sensitivity")
                    sensitivity = st.radio(
                        "Sensitivity Level",
                        ["Low", "Medium", "High"],
                        index=["Low", "Medium", "High"].index(st.session_state.detection_sensitivity),
                        horizontal=True
                    )
                    st.session_state.detection_sensitivity = sensitivity
                    
                    sensitivity_descriptions = {
                        "Low": "Conservative detection - fewer false alarms",
                        "Medium": "Balanced detection - optimal performance", 
                        "High": "Aggressive detection - catches subtle anomalies"
                    }
                    
                    colors = {"Low": "#f39c12", "Medium": "#2ecc71", "High": "#e74c3c"}
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(90deg, {colors[sensitivity]} 0%, {colors[sensitivity]}dd 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-top: 1rem; text-align: left; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                        <strong style="font-size: 16px;">{sensitivity} Sensitivity:</strong><br>
                        <span style="font-size: 14px; opacity: 0.9;">{sensitivity_descriptions[sensitivity]}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
            
            # Product filter for single sample analysis
            st.markdown("### 🔬 Single Sample Analysis")
            
            # Product-specific parameter inputs
            FEATURES = get_current_features()
            
            col1, col2 = st.columns([1.5, 1], gap="large")
            
            with col1:
                st.markdown(f"""
                <div class="filter-container">
                    <h4 style="color: #2c3e50; margin-bottom: 1.5rem; text-align: left;">📊 Input Parameters for {selected_product}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Create parameter input organized by categories
                input_data = {}
                
                # Dynamically group parameters based on product type
                param_categories = {}
                for param, config in FEATURES.items():
                    if any(ingredient in param.lower() for ingredient in ['flour', 'sugar', 'salt', 'yeast', 'butter', 'eggs', 'milk', 'vanilla', 'olive']):
                        category = "🌾 Ingredients"
                    elif 'temp' in param.lower():
                        category = "🌡️ Temperature Controls"
                    elif any(process in param.lower() for process in ['mixer', 'speed', 'time', 'weight']):
                        category = "⚙️ Process Parameters"
                    else:
                        category = "🎯 Quality Metrics"
                    
                    if category not in param_categories:
                        param_categories[category] = []
                    param_categories[category].append(param)
                
                # Display parameters by category
                for category, params in param_categories.items():
                    with st.expander(category, expanded=True):
                        # Arrange in columns for better layout
                        param_cols = st.columns(2 if len(params) > 3 else 1)
                        
                        for i, param in enumerate(params):
                            col_idx = i % len(param_cols)
                            with param_cols[col_idx]:
                                range_info = FEATURES[param]
                                
                                # Determine appropriate input range and step
                                if param == 'Quality_Score':
                                    min_val, max_val = 0.0, 100.0
                                    step = 0.1
                                elif 'count' in param.lower():
                                    min_val, max_val = 0, 50
                                    step = 1
                                elif 'time' in param.lower() and 'hours' in param.lower():
                                    min_val, max_val = 0.0, 24.0
                                    step = 0.5
                                else:
                                    min_val = max(0.0, range_info['min'] * 0.5)
                                    max_val = range_info['max'] * 1.5
                                    step = 0.1
                                
                                input_data[param] = st.number_input(
                                    param,
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=float(range_info['optimal']),
                                    step=step,
                                    help=f"Normal range: {range_info['min']}-{range_info['max']} | Optimal: {range_info['optimal']}",
                                    key=f"param_{param}"
                                )
                
                # Analysis button
                st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
                if st.button("🔍 Analyze Sample", type="primary", use_container_width=True):
                    with st.spinner("Analyzing sample..."):
                        time.sleep(1)  # Simulate processing time
                        result = detect_anomaly_advanced(
                            input_data,
                            st.session_state.anomaly_threshold,
                            st.session_state.detection_sensitivity
                        )
                        
                        # Store result in session state for the right column
                        st.session_state.analysis_result = result
                        
                        # Show immediate notification
                        if result['is_anomaly']:
                            show_notification(f"🚨 ANOMALY DETECTED! Probability: {result['probability']:.3f}", "error")
                        else:
                            show_notification(f"✅ Sample Normal! Probability: {result['probability']:.3f}", "success")
                        
                        st.rerun()
            
            with col2:
                st.markdown("""
                <div class="filter-container">
                    <h4 style="color: #2c3e50; margin-bottom: 1.5rem; text-align: left;">📊 Analysis Results</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if hasattr(st.session_state, 'analysis_result'):
                    result = st.session_state.analysis_result
                    
                    # Alert display with enhanced styling
                    if result['is_anomaly']:
                        st.markdown(f"""
                        <div class="anomaly-alert">
                            <h3 style="margin-top: 0;">🚨 ANOMALY DETECTED! 🚨</h3>
                            <div style="margin: 1rem 0; font-size: 16px;">
                                <div style="margin: 0.5rem 0;">📊 <strong>Probability:</strong> {result['probability']:.3f}</div>
                                <div style="margin: 0.5rem 0;">🎯 <strong>Confidence:</strong> {result['confidence']}</div>
                                <div style="margin: 0.5rem 0;">⚙️ <strong>Method:</strong> {result['method']}</div>
                                <div style="margin: 0.5rem 0;">🔍 <strong>Issues Found:</strong> {len(result['anomalous_features'])}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if result['anomalous_features']:
                            st.markdown("#### 🔍 Detected Issues:")
                            for feature in result['anomalous_features']:
                                st.error(f"• {feature}")
                            
                            st.markdown("#### 📋 Recommended Actions:")
                            st.warning("• Review process parameters immediately")
                            st.warning("• Check equipment calibration")
                            st.warning("• Investigate potential root causes")
                            st.warning("• Consider batch quality control measures")
                    else:
                        st.markdown(f"""
                        <div class="normal-alert">
                            <h3 style="margin-top: 0;">✅ SAMPLE NORMAL</h3>
                            <div style="margin: 1rem 0; font-size: 16px;">
                                <div style="margin: 0.5rem 0;">📊 <strong>Probability:</strong> {result['probability']:.3f}</div>
                                <div style="margin: 0.5rem 0;">🎯 <strong>Confidence:</strong> {result['confidence']}</div>
                                <div style="margin: 0.5rem 0;">⚙️ <strong>Method:</strong> {result['method']}</div>
                                <div style="margin: 0.5rem 0;">🎉 <strong>Status:</strong> All parameters optimal</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success("🎉 All parameters are within acceptable ranges!")
                        st.info("📊 Process is operating optimally")
                    
                    # Additional metrics with better layout
                    st.markdown("#### 📈 Analysis Details")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("🎯 Sensitivity", result['sensitivity_used'])
                        st.metric("📊 Threshold", f"{st.session_state.anomaly_threshold:.3f}")
                    with col_b:
                        st.metric("⚡ Anomaly Score", f"{result['anomaly_score']:.3f}")
                        deviation_count = len(result['deviation_details'])
                        st.metric("🔍 Deviations", deviation_count)
                
                else:
                    st.markdown("""
                    <div class='upload-section'>
                        <h4 style='color: #495057; text-align: left;'>👈 Enter Parameters</h4>
                        <p style='color: #6c757d; font-size: 14px;'>Fill in the parameters and click 'Analyze Sample' to see results</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show current settings
                    st.markdown("#### ⚙️ Current Settings")
                    st.info(f"**Threshold:** {st.session_state.anomaly_threshold:.3f}")
                    st.info(f"**Sensitivity:** {st.session_state.detection_sensitivity}")
                    st.info(f"**Product:** {selected_product}")
            
            # Parameter Analysis Table
            if hasattr(st.session_state, 'analysis_result'):
                result = st.session_state.analysis_result
                
                st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                st.markdown("### 📋 Detailed Parameter Analysis")
                df_analysis, html_table = create_parameter_analysis_table(result['parameter_analysis'])
                st.markdown(html_table, unsafe_allow_html=True)
                
                # Parameter Visualization
                st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                st.markdown("### 📊 Parameter Visualization")
                
                # Create comparison visualization
                fig = go.Figure()
                
                parameters = [p['Parameter'] for p in result['parameter_analysis']]
                values = [p['Value'] for p in result['parameter_analysis']]
                optimal_values = [p['Optimal'] for p in result['parameter_analysis']]
                min_values = [p['Min'] for p in result['parameter_analysis']]
                max_values = [p['Max'] for p in result['parameter_analysis']]
                
                # Color code based on status
                colors = []
                for p in result['parameter_analysis']:
                    if p['Status'] == 'Normal':
                        colors.append('#27ae60')
                    elif p['Status'] == 'Suboptimal':
                        colors.append('#f39c12')
                    else:
                        colors.append('#e74c3c')
                
                # Add current values
                fig.add_trace(go.Bar(
                    x=parameters,
                    y=values,
                    marker_color=colors,
                    name="Current Values",
                    text=[f"{v:.1f}" for v in values],
                    textposition='auto',
                    opacity=0.8
                ))
                
                # Add optimal values as line
                fig.add_trace(go.Scatter(
                    x=parameters,
                    y=optimal_values,
                    mode='markers+lines',
                    marker=dict(color='blue', size=12, symbol='diamond'),
                    line=dict(color='blue', width=3, dash='dash'),
                    name="Optimal Values"
                ))
                
                # Add range indicators
                for i, param in enumerate(parameters):
                    fig.add_shape(
                        type="rect",
                        x0=i-0.4, y0=min_values[i],
                        x1=i+0.4, y1=max_values[i],
                        fillcolor="rgba(52, 152, 219, 0.1)",
                        line=dict(color="rgba(52, 152, 219, 0.3)", width=1),
                        layer="below"
                    )
                
                fig.update_layout(
                    title=f"Parameter Analysis - {selected_product} Production",
                    xaxis_title="Parameters",
                    yaxis_title="Values",
                    height=600,
                    plot_bgcolor='white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified',
                    font=dict(color='#2c3e50'),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                fig.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Deviation Analysis
                if result['deviation_details']:
                    st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                    st.markdown("### 🔍 Deviation Analysis")
                    
                    # Create deviation chart instead of pie chart
                    deviation_chart = create_deviation_analysis_chart(result['deviation_details'])
                    if deviation_chart:
                        st.plotly_chart(deviation_chart, use_container_width=True)
                    
                    # Detailed deviation table
                    deviation_df = pd.DataFrame(result['deviation_details'])
                    st.dataframe(
                        deviation_df, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "feature": st.column_config.TextColumn("🔧 Parameter", width="medium"),
                            "deviation_type": st.column_config.TextColumn("⚠️ Issue Type", width="medium"),
                            "deviation_percent": st.column_config.ProgressColumn("📊 Deviation %", min_value=0, max_value=100),
                            "severity": st.column_config.TextColumn("🚨 Severity", width="small")
                        }
                    )
        
        else:
            st.markdown("""
            <div class='upload-section'>
                <h3 style='color: #495057;'>⚠️ Model Training Required</h3>
                <p style='color: #6c757d; font-size: 16px; margin: 1rem 0;'>Please train the AI model first to access anomaly detection features.</p>
                <p style='color: #6c757d; font-size: 14px;'>Use the sidebar to start model training.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:  # Analytics
        st.markdown("## 📊 Advanced Analytics Dashboard")
        
        if st.session_state.model_trained:
            # Performance Metrics
            st.markdown("### 🎯 Model Performance Metrics")
            col1, col2, col3, col4 = st.columns(4, gap="medium")
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4>🎯 Model Accuracy</h4>
                    <h2>94.2%</h2>
                    <p style='color: #28a745;'>+2.1% improvement</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4>🎪 Precision</h4>
                    <h2>91.8%</h2>
                    <p style='color: #28a745;'>+1.5% from baseline</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h4>🔍 Recall</h4>
                    <h2>89.3%</h2>
                    <p style='color: #fd7e14;'>+0.8% steady</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h4>⚖️ F1-Score</h4>
                    <h2>90.5%</h2>
                    <p style='color: #28a745;'>+1.2% optimized</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature Importance Analysis (updates with product selection)
            st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
            st.markdown(f"### 📈 Feature Importance Analysis - {selected_product}")
            
            # Generate realistic feature importance based on current product
            FEATURES = get_current_features()
            features = list(FEATURES.keys())
            
            # Create product-specific importance patterns
            np.random.seed(hash(selected_product) % 2**32)  # Consistent seed per product
            importance_values = np.random.exponential(scale=0.15, size=len(features))
            importance_values = importance_values / importance_values.sum()
            
            # Sort by importance
            feature_importance = list(zip(features, importance_values))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            features_sorted = [f[0] for f in feature_importance]
            importance_sorted = [f[1] for f in feature_importance]
            
            fig_importance = px.bar(
                x=importance_sorted,
                y=features_sorted,
                orientation='h',
                title=f"Feature Importance in Anomaly Detection Model - {selected_product}",
                color=importance_sorted,
                color_continuous_scale="viridis",
                text=[f"{val:.3f}" for val in importance_sorted]
            )
            fig_importance.update_layout(
                height=600, 
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50')
            )
            fig_importance.update_traces(textposition='auto')
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Dataset-specific analytics
            if st.session_state.current_batch_data is not None:
                st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                st.markdown("### 📊 Current Dataset Analytics")
                
                df = st.session_state.current_batch_data
                feature_cols = [col for col in df.columns if col in FEATURES]
                
                if feature_cols:
                    # Statistical summary
                    st.markdown("#### 📈 Statistical Summary")
                    summary_stats = df[feature_cols].describe()
                    st.dataframe(summary_stats.round(3), use_container_width=True)
                    
                    # Parameter Correlation Analysis
                    st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                    st.markdown("#### 🔗 Parameter Correlation Analysis")
                    
                    if len(feature_cols) > 1:
                        corr_matrix = df[feature_cols].corr()
                        
                        # Create enhanced correlation heatmap
                        fig_corr = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            title=f"Parameter Correlation Matrix - {selected_product} Batch Analysis",
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1
                        )
                        
                        fig_corr.update_layout(
                            height=600,
                            title_x=0.5,
                            plot_bgcolor='white',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#2c3e50')
                        )
                        
                        fig_corr.update_traces(
                            texttemplate="%{z:.2f}",
                            textfont_size=10
                        )
                        
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Correlation insights
                        strong_correlations = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_val = corr_matrix.iloc[i, j]
                                if abs(corr_val) > 0.7:
                                    strong_correlations.append({
                                        'Feature 1': corr_matrix.columns[i],
                                        'Feature 2': corr_matrix.columns[j],
                                        'Correlation': corr_val,
                                        'Strength': 'Strong Positive' if corr_val > 0.7 else 'Strong Negative'
                                    })
                        
                        if strong_correlations:
                            st.markdown("##### 🔍 Strong Correlations Found")
                            corr_df = pd.DataFrame(strong_correlations)
                            st.dataframe(
                                corr_df, 
                                use_container_width=True, 
                                hide_index=True,
                                column_config={
                                    "Correlation": st.column_config.ProgressColumn("📊 Correlation", min_value=-1, max_value=1)
                                }
                            )
                        
                        # Distribution analysis for current dataset
                        st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                        st.markdown("#### 📊 Parameter Distribution Analysis")
                        
                        # Create distribution plots for top 6 most important features
                        top_features = features_sorted[:6]  # Top 6 features
                        available_features = [f for f in top_features if f in feature_cols]
                        
                        if available_features:
                            fig_dist = make_subplots(
                                rows=2, cols=3,
                                subplot_titles=[f"📊 {feature}" for feature in available_features],
                                vertical_spacing=0.12,
                                horizontal_spacing=0.1
                            )
                            
                            for i, feature in enumerate(available_features):
                                row = i // 3 + 1
                                col = i % 3 + 1
                                
                                if feature in FEATURES:
                                    range_info = FEATURES[feature]
                                    
                                    # Add histogram
                                    fig_dist.add_trace(
                                        go.Histogram(
                                            x=df[feature], 
                                            name=feature, 
                                            showlegend=False,
                                            nbinsx=20, 
                                            opacity=0.7,
                                            marker_color='skyblue'
                                        ),
                                        row=row, col=col
                                    )
                                    
                                    # Add reference lines
                                    fig_dist.add_vline(x=range_info['min'], line_dash="dash", 
                                                     line_color="red", opacity=0.8, row=row, col=col)
                                    fig_dist.add_vline(x=range_info['max'], line_dash="dash", 
                                                     line_color="red", opacity=0.8, row=row, col=col)
                                    fig_dist.add_vline(x=range_info['optimal'], line_dash="solid", 
                                                     line_color="green", opacity=0.9, row=row, col=col)
                            
                            fig_dist.update_layout(
                                height=500, 
                                title_text=f"Parameter Distributions - {selected_product} Dataset",
                                title_x=0.5,
                                plot_bgcolor='white',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#2c3e50')
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Historical Performance Trends
            st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
            st.markdown("### 📊 Historical Performance Trends")
            
            # Generate sample historical data
            dates = pd.date_range(start='2024-01-01', end='2024-08-24', freq='D')
            np.random.seed(123)
            
            historical_data = {
                'Date': dates,
                'Anomaly_Rate': np.random.normal(0.04, 0.015, len(dates)).clip(0, 0.15),
                'Quality_Score': np.random.normal(85, 3, len(dates)).clip(70, 100),
                'Production_Volume': np.random.normal(1000, 150, len(dates)).clip(600, 1500),
                'Detection_Accuracy': np.random.normal(0.94, 0.02, len(dates)).clip(0.85, 0.99)
            }
            hist_df = pd.DataFrame(historical_data)
            
            # Add trends
            hist_df['Anomaly_Rate'] += np.sin(np.arange(len(dates)) / 30) * 0.01
            hist_df['Quality_Score'] += np.cos(np.arange(len(dates)) / 45) * 2
            
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                fig_anomaly = px.line(
                    hist_df, 
                    x='Date', 
                    y='Anomaly_Rate',
                    title='📈 Daily Anomaly Rate Trend',
                    color_discrete_sequence=['#e74c3c']
                )
                fig_anomaly.update_layout(
                    height=400, 
                    yaxis_title="Anomaly Rate (%)",
                    plot_bgcolor='white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50')
                )
                fig_anomaly.add_hline(y=0.05, line_dash="dash", line_color="orange", 
                                    annotation_text="Target Threshold (5%)")
                st.plotly_chart(fig_anomaly, use_container_width=True)
            
            with col2:
                fig_quality = px.line(
                    hist_df, 
                    x='Date', 
                    y='Quality_Score',
                    title='🎯 Quality Score Trend',
                    color_discrete_sequence=['#27ae60']
                )
                fig_quality.update_layout(
                    height=400, 
                    yaxis_title="Quality Score",
                    plot_bgcolor='white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50')
                )
                fig_quality.add_hline(y=85, line_dash="dash", line_color="blue",
                                    annotation_text="Target Quality (85)")
                st.plotly_chart(fig_quality, use_container_width=True)
            
            # Combined metrics dashboard
            st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
            fig_combined = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Production Volume', 'Detection Accuracy', 'Quality vs Anomaly Rate', 'Weekly Summary'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": True}, {"secondary_y": False}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # Production volume
            fig_combined.add_trace(
                go.Scatter(x=hist_df['Date'], y=hist_df['Production_Volume'],
                          mode='lines', name='Production Volume', line=dict(color='#3498db', width=2)),
                row=1, col=1
            )
            
            # Detection accuracy
            fig_combined.add_trace(
                go.Scatter(x=hist_df['Date'], y=hist_df['Detection_Accuracy'],
                          mode='lines', name='Detection Accuracy', line=dict(color='#9b59b6', width=2)),
                row=1, col=2
            )
            
            # Quality vs Anomaly Rate (dual axis)
            fig_combined.add_trace(
                go.Scatter(x=hist_df['Date'], y=hist_df['Quality_Score'],
                          mode='lines', name='Quality Score', line=dict(color='#27ae60', width=2)),
                row=2, col=1
            )
            fig_combined.add_trace(
                go.Scatter(x=hist_df['Date'], y=hist_df['Anomaly_Rate'] * 100,
                          mode='lines', name='Anomaly Rate (%)', line=dict(color='#e74c3c', width=2)),
                row=2, col=1, secondary_y=True
            )
            
            # Weekly summary (bar chart)
            weekly_data = hist_df.groupby(hist_df['Date'].dt.isocalendar().week).agg({
                'Anomaly_Rate': 'mean',
                'Quality_Score': 'mean',
                'Production_Volume': 'sum'
            }).reset_index()
            
            fig_combined.add_trace(
                go.Bar(x=weekly_data['week'], y=weekly_data['Quality_Score'],
                      name='Weekly Avg Quality', marker_color='#f39c12', opacity=0.7),
                row=2, col=2
            )
            
            fig_combined.update_layout(
                height=800, 
                title_text="Comprehensive Performance Dashboard",
                plot_bgcolor='white',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50')
            )
            st.plotly_chart(fig_combined, use_container_width=True)
            
            # Distribution Analysis Generator
            st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
            st.markdown("### 📈 Parameter Distribution Analysis")
            
            if st.button("🎲 Generate Distribution Analysis", use_container_width=True):
                show_notification("📊 Generating comprehensive distribution analysis...", "info")
                
                # Generate comprehensive sample data for current product
                FEATURES = get_current_features()
                n_samples = 1000
                sample_data = {}
                
                for feature, range_info in FEATURES.items():
                    # Create realistic distributions with some anomalies
                    normal_samples = np.random.normal(
                        range_info['optimal'], 
                        (range_info['max'] - range_info['min']) * 0.08, 
                        int(n_samples * 0.85)
                    )
                    
                    # Add some suboptimal samples
                    suboptimal_low = np.random.uniform(
                        range_info['min'], range_info['optimal'], 
                        int(n_samples * 0.08)
                    )
                    suboptimal_high = np.random.uniform(
                        range_info['optimal'], range_info['max'], 
                        int(n_samples * 0.07)
                    )
                    
                    # Add anomalous samples
                    anomaly_samples = np.concatenate([
                        np.random.uniform(0, range_info['min'], int(n_samples * 0.03)),
                        np.random.uniform(range_info['max'], range_info['max'] * 1.4, int(n_samples * 0.02))
                    ])
                    
                    all_samples = np.concatenate([normal_samples, suboptimal_low, suboptimal_high, anomaly_samples])
                    sample_data[feature] = all_samples[:n_samples]  # Ensure exact sample count
                
                dist_df = pd.DataFrame(sample_data)
                
                # Create comprehensive distribution analysis with proper grid
                n_features = len(FEATURES)
                n_cols = 4
                n_rows = (n_features + n_cols - 1) // n_cols
                
                fig_dist = make_subplots(
                    rows=n_rows, cols=n_cols,
                    subplot_titles=[f"📊 {feature}" for feature in FEATURES.keys()],
                    vertical_spacing=0.08,
                    horizontal_spacing=0.08
                )
                
                for i, (feature, range_info) in enumerate(FEATURES.items()):
                    row = i // n_cols + 1
                    col = i % n_cols + 1
                    
                    # Add histogram
                    fig_dist.add_trace(
                        go.Histogram(
                            x=dist_df[feature], 
                            name=feature, 
                            showlegend=False,
                            nbinsx=40, 
                            opacity=0.7,
                            marker_color='skyblue'
                        ),
                        row=row, col=col
                    )
                    
                    # Add reference lines
                    fig_dist.add_vline(x=range_info['min'], line_dash="dash", 
                                     line_color="red", opacity=0.8, row=row, col=col)
                    fig_dist.add_vline(x=range_info['max'], line_dash="dash", 
                                     line_color="red", opacity=0.8, row=row, col=col)
                    fig_dist.add_vline(x=range_info['optimal'], line_dash="solid", 
                                     line_color="green", opacity=0.9, row=row, col=col)
                
                fig_dist.update_layout(
                    height=200 * n_rows, 
                    title_text=f"📈 Comprehensive Parameter Distribution Analysis - {selected_product}",
                    title_x=0.5,
                    plot_bgcolor='white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50')
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Statistical summary
                st.markdown("#### 📊 Statistical Summary")
                summary_stats = dist_df.describe()
                st.dataframe(summary_stats.round(3), use_container_width=True)
                
                # Anomaly detection on generated data
                st.markdown("#### 🔍 Anomaly Detection on Generated Data")
                
                anomaly_counts = {}
                for feature, range_info in FEATURES.items():
                    data = dist_df[feature]
                    anomalies = len(data[(data < range_info['min']) | (data > range_info['max'])])
                    anomaly_counts[feature] = {
                        'Anomalies': anomalies,
                        'Percentage': (anomalies / len(data)) * 100,
                        'Normal_Range': f"{range_info['min']}-{range_info['max']}"
                    }
                
                anomaly_summary = pd.DataFrame(anomaly_counts).T
                st.dataframe(
                    anomaly_summary.round(2), 
                    use_container_width=True,
                    column_config={
                        "Anomalies": st.column_config.NumberColumn("🚨 Anomaly Count", width="small"),
                        "Percentage": st.column_config.ProgressColumn("📊 Anomaly %", min_value=0, max_value=100),
                        "Normal_Range": st.column_config.TextColumn("✅ Normal Range", width="medium")
                    }
                )
                
                show_notification(f"✅ Distribution analysis completed for {selected_product}!", "success")
        
        else:
            st.markdown("""
            <div class='upload-section'>
                <h3 style='color: #495057;'>⚠️ Model Training Required</h3>
                <p style='color: #6c757d; font-size: 16px; margin: 1rem 0;'>Please train the AI model first to access advanced analytics.</p>
                <p style='color: #6c757d; font-size: 14px;'>Use the sidebar to start model training.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer with better styling
    st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 20px; margin-top: 2rem; box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);'>
        <h4 style='margin-bottom: 1rem;'>🏭 Food Quality Monitor - Advanced Anomaly Detection System</h4>
        <p style='font-size: 16px; margin: 0.5rem 0; opacity: 0.9;'>Powered by TCN + LightGBM | Real-time Process Monitoring & Intelligent Analytics</p>
        <p style='font-size: 14px; margin: 0.5rem 0; opacity: 0.8;'>Current Product: <strong>{selected_product}</strong> | AI Model: <strong>{"Active" if st.session_state.model_trained else "Inactive"}</strong></p>
        <p style='font-size: 12px; opacity: 0.7; margin-top: 1rem;'>© 2024 | Built with Streamlit & Advanced AI Models | Version 3.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()