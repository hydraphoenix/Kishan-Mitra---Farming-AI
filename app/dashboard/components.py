"""
Streamlit Dashboard Components for AgroMRV System
Reusable UI components for professional agricultural MRV dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardComponents:
    """Professional UI components for AgroMRV dashboard"""
    
    @staticmethod
    def page_header(title: str, subtitle: str = "", icon: str = "🌱"):
        """Create professional page header with gradient styling"""
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #2E8B57 0%, #3CB371 50%, #20B2AA 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h1 style="color: white; margin: 0; text-align: center; font-size: 2.5rem;">
                {icon} {title}
            </h1>
            {f'<p style="color: #E8F5E8; text-align: center; font-size: 1.2rem; margin: 0.5rem 0 0 0;">{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def metrics_row(metrics: List[Dict], columns: int = 4):
        """Create a row of metric cards"""
        cols = st.columns(columns)
        
        for i, metric in enumerate(metrics[:columns]):
            with cols[i % columns]:
                DashboardComponents.metric_card(
                    metric['label'],
                    metric['value'],
                    metric.get('delta', None),
                    metric.get('color', 'green')
                )
    
    @staticmethod
    def metric_card(label: str, value: Any, delta: Optional[str] = None, 
                   color: str = "green", help_text: str = ""):
        """Create professional metric card with styling"""
        
        # Color mapping
        color_map = {
            'green': '#2E8B57',
            'blue': '#4682B4',
            'orange': '#FF8C00',
            'red': '#DC143C',
            'purple': '#9370DB'
        }
        
        card_color = color_map.get(color, color)
        
        # Format value
        if isinstance(value, (int, float)):
            if value >= 1000:
                formatted_value = f"{value:,.0f}" if isinstance(value, int) else f"{value:,.2f}"
            else:
                formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
        else:
            formatted_value = str(value)
        
        delta_html = ""
        if delta:
            delta_color = "#2E8B57" if not delta.startswith("-") else "#DC143C"
            delta_html = f'<p style="color: {delta_color}; font-size: 0.9rem; margin: 0;">{delta}</p>'
        
        st.markdown(f"""
        <div style="
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid {card_color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        ">
            <p style="color: #666; font-size: 0.9rem; margin: 0; font-weight: 500;">{label}</p>
            <h3 style="color: {card_color}; margin: 0.5rem 0; font-size: 1.8rem;">{formatted_value}</h3>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
        
        if help_text:
            st.caption(help_text)
    
    @staticmethod
    def info_card(title: str, content: str, icon: str = "ℹ️", color: str = "blue"):
        """Create informational card"""
        color_map = {
            'blue': '#E3F2FD',
            'green': '#E8F5E8',
            'yellow': '#FFFBF0',
            'red': '#FFEBEE'
        }
        
        bg_color = color_map.get(color, color)
        
        st.markdown(f"""
        <div style="
            background: {bg_color};
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            border: 1px solid rgba(0,0,0,0.1);
        ">
            <h4 style="margin: 0 0 1rem 0; color: #333;">
                {icon} {title}
            </h4>
            <p style="margin: 0; color: #555; line-height: 1.6;">{content}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def status_badge(status: str, label: str = ""):
        """Create status badge with color coding"""
        status_colors = {
            'active': '#28A745',
            'success': '#28A745',
            'verified': '#17A2B8',
            'pending': '#FFC107',
            'warning': '#FF8C00',
            'error': '#DC3545',
            'failed': '#DC3545'
        }
        
        color = status_colors.get(status.lower(), '#6C757D')
        display_text = label if label else status.title()
        
        st.markdown(f"""
        <span style="
            background: {color};
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
        ">{display_text}</span>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def progress_bar(value: float, max_value: float = 100, 
                    label: str = "", color: str = "#2E8B57"):
        """Create custom progress bar"""
        percentage = min(100, (value / max_value) * 100)
        
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            {f'<p style="margin: 0 0 0.5rem 0; font-weight: 500;">{label}</p>' if label else ''}
            <div style="
                background: #E0E0E0;
                border-radius: 10px;
                height: 20px;
                overflow: hidden;
            ">
                <div style="
                    background: {color};
                    height: 100%;
                    width: {percentage}%;
                    border-radius: 10px;
                    transition: width 0.3s ease;
                "></div>
            </div>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.9rem; color: #666;">
                {value:.1f} / {max_value} ({percentage:.1f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def data_table(df: pd.DataFrame, title: str = "", 
                  max_rows: int = 10, searchable: bool = True):
        """Create professional data table with search"""
        if title:
            st.subheader(title)
        
        if searchable and len(df) > 5:
            search_term = st.text_input("🔍 Search table:", key=f"search_{title}")
            if search_term:
                # Search across all string columns
                string_cols = df.select_dtypes(include=['object']).columns
                mask = df[string_cols].astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                df = df[mask]
        
        # Display table with custom styling
        if len(df) > max_rows:
            st.write(f"Showing first {max_rows} rows of {len(df)} total:")
            st.dataframe(
                df.head(max_rows),
                use_container_width=True,
                height=400
            )
        else:
            st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def farm_selector(farms: List[Dict], key: str = "farm_select") -> Optional[Dict]:
        """Create farm selection dropdown"""
        if not farms:
            st.warning("No farms available")
            return None
        
        farm_options = [f"{farm['farm_id']} - {farm['state']} ({farm['crop_type']})" 
                       for farm in farms]
        
        selected_idx = st.selectbox(
            "Select Farm:",
            range(len(farm_options)),
            format_func=lambda x: farm_options[x],
            key=key
        )
        
        return farms[selected_idx] if selected_idx is not None else None
    
    @staticmethod
    def filter_panel(df: pd.DataFrame, columns_to_filter: List[str]) -> pd.DataFrame:
        """Create comprehensive filter panel for data"""
        st.sidebar.markdown("### 🔧 Data Filters")
        
        filtered_df = df.copy()
        
        for column in columns_to_filter:
            if column in df.columns:
                if df[column].dtype in ['object']:
                    # Categorical filter
                    unique_values = sorted(df[column].unique())
                    selected_values = st.sidebar.multiselect(
                        f"Filter by {column}:",
                        options=unique_values,
                        default=unique_values,
                        key=f"filter_{column}"
                    )
                    if selected_values:
                        filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
                
                elif df[column].dtype in ['int64', 'float64']:
                    # Numerical range filter
                    min_val = float(df[column].min())
                    max_val = float(df[column].max())
                    
                    if min_val < max_val:
                        selected_range = st.sidebar.slider(
                            f"Filter by {column}:",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key=f"filter_{column}"
                        )
                        filtered_df = filtered_df[
                            (filtered_df[column] >= selected_range[0]) & 
                            (filtered_df[column] <= selected_range[1])
                        ]
        
        # Date filter if date column exists
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                min_date = df['date'].min().date()
                max_date = df['date'].max().date()
                
                date_range = st.sidebar.date_input(
                    "Date Range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="date_filter"
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[
                        (filtered_df['date'].dt.date >= start_date) &
                        (filtered_df['date'].dt.date <= end_date)
                    ]
            except:
                st.sidebar.warning("Could not parse date column for filtering")
        
        # Show filter summary
        if len(filtered_df) != len(df):
            st.sidebar.info(f"Showing {len(filtered_df):,} of {len(df):,} records")
        
        return filtered_df
    
    @staticmethod
    def export_section(data: Any, filename_base: str = "agromrv_export"):
        """Create export options section"""
        st.markdown("### 📥 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with col1:
            if st.button("📊 Export CSV", use_container_width=True):
                if isinstance(data, pd.DataFrame):
                    csv_data = data.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        file_name=f"{filename_base}_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.error("Data must be a DataFrame for CSV export")
        
        with col2:
            if st.button("📄 Export JSON", use_container_width=True):
                if isinstance(data, pd.DataFrame):
                    json_data = data.to_json(orient='records', indent=2)
                else:
                    import json
                    json_data = json.dumps(data, indent=2, default=str)
                
                st.download_button(
                    "Download JSON",
                    json_data,
                    file_name=f"{filename_base}_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col3:
            if st.button("📑 Export Report", use_container_width=True):
                # Generate summary report
                if isinstance(data, pd.DataFrame):
                    report = {
                        'export_timestamp': datetime.now().isoformat(),
                        'total_records': len(data),
                        'columns': list(data.columns),
                        'summary_statistics': data.describe().to_dict() if len(data) > 0 else {}
                    }
                else:
                    report = {
                        'export_timestamp': datetime.now().isoformat(),
                        'data_type': type(data).__name__,
                        'data': data
                    }
                
                import json
                report_data = json.dumps(report, indent=2, default=str)
                st.download_button(
                    "Download Report",
                    report_data,
                    file_name=f"report_{filename_base}_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    @staticmethod
    def loading_spinner(text: str = "Processing..."):
        """Create loading spinner with custom text"""
        return st.spinner(text)
    
    @staticmethod
    def alert(message: str, alert_type: str = "info"):
        """Create styled alert boxes"""
        if alert_type == "success":
            st.success(message)
        elif alert_type == "warning":
            st.warning(message)
        elif alert_type == "error":
            st.error(message)
        else:
            st.info(message)
    
    @staticmethod
    def collapsible_section(title: str, content_func, expanded: bool = False):
        """Create collapsible section with content"""
        with st.expander(title, expanded=expanded):
            content_func()
    
    @staticmethod
    def demo_mode_toggle():
        """Create demo mode toggle with sample data"""
        demo_mode = st.sidebar.checkbox(
            "🎯 Demo Mode",
            value=True,
            help="Use pre-filled sample data for demonstration"
        )
        
        if demo_mode:
            st.sidebar.success("Demo mode active - using sample data")
        
        return demo_mode
    
    @staticmethod
    def nabard_branding():
        """Add NABARD Hackathon 2025 branding"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 1rem; background: #F0F8F0; border-radius: 8px;">
            <h4 style="color: #2E8B57; margin: 0;">🏆 NABARD Hackathon 2025</h4>
            <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">
                Climate Finance for Smallholder Farmers
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def system_status_panel():
        """Create system status monitoring panel"""
        st.sidebar.markdown("### 🔍 System Status")
        
        status_items = [
            {"label": "AI Models", "status": "active", "icon": "🤖"},
            {"label": "Blockchain", "status": "verified", "icon": "⛓️"},
            {"label": "IPCC Compliance", "status": "verified", "icon": "✅"},
            {"label": "Data Quality", "status": "active", "icon": "📊"}
        ]
        
        for item in status_items:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(f"{item['icon']} {item['label']}")
            with col2:
                DashboardComponents.status_badge(item['status'])
    
    @staticmethod
    def quick_stats_sidebar(data: pd.DataFrame):
        """Create quick statistics in sidebar"""
        if data is None or data.empty:
            return
        
        st.sidebar.markdown("### 📈 Quick Stats")
        
        # Basic statistics
        st.sidebar.metric("Total Records", f"{len(data):,}")
        
        if 'farm_id' in data.columns:
            st.sidebar.metric("Unique Farms", data['farm_id'].nunique())
        
        if 'state' in data.columns:
            st.sidebar.metric("States Covered", data['state'].nunique())
        
        if 'carbon_credits_potential' in data.columns:
            total_credits = data['carbon_credits_potential'].sum()
            st.sidebar.metric("Total Carbon Credits", f"{total_credits:.3f}")
        
        # Date range
        if 'date' in data.columns:
            try:
                date_range = pd.to_datetime(data['date'])
                st.sidebar.metric("Data Range (Days)", (date_range.max() - date_range.min()).days + 1)
            except:
                pass

class FormComponents:
    """Form components for data input"""
    
    @staticmethod
    def farm_input_form(key_prefix: str = "") -> Dict:
        """Create comprehensive farm data input form"""
        st.markdown("### 🌾 Farm Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            farm_id = st.text_input(
                "Farm ID",
                value=f"FARM{np.random.randint(1000, 9999)}",
                key=f"{key_prefix}farm_id"
            )
            
            state = st.selectbox(
                "State",
                ['Punjab', 'Haryana', 'Uttar Pradesh', 'Bihar', 'West Bengal',
                 'Madhya Pradesh', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Kerala',
                 'Andhra Pradesh', 'Telangana', 'Gujarat', 'Rajasthan', 'Odisha'],
                key=f"{key_prefix}state"
            )
            
            crop_type = st.selectbox(
                "Crop Type",
                ['rice', 'wheat', 'maize', 'vegetables', 'millet', 'sugarcane', 'cotton', 'pulses'],
                key=f"{key_prefix}crop_type"
            )
        
        with col2:
            area_hectares = st.number_input(
                "Farm Area (hectares)",
                min_value=0.1,
                max_value=50.0,
                value=2.5,
                step=0.1,
                key=f"{key_prefix}area"
            )
            
            management_practice = st.selectbox(
                "Management Practice",
                ['conventional', 'organic', 'reduced_tillage', 'no_tillage'],
                key=f"{key_prefix}management"
            )
            
            mechanization_level = st.selectbox(
                "Mechanization Level",
                ['low', 'medium', 'high'],
                index=1,
                key=f"{key_prefix}mechanization"
            )
        
        return {
            'farm_id': farm_id,
            'state': state,
            'crop_type': crop_type,
            'area_hectares': area_hectares,
            'management_practice': management_practice,
            'mechanization_level': mechanization_level
        }
    
    @staticmethod
    def environmental_input_form(key_prefix: str = "", enable_weather_api: bool = True) -> Dict:
        """Create environmental parameters input form with live weather integration"""
        st.markdown("### 🌡️ Environmental Parameters")
        
        # Weather API integration
        if enable_weather_api:
            col_weather1, col_weather2 = st.columns([3, 1])
            with col_weather1:
                st.info("💡 Weather data can be auto-filled from live API")
            with col_weather2:
                if st.button("🌦️ Get Live Weather", key=f"{key_prefix}weather_btn"):
                    try:
                        # Import weather API
                        import sys
                        import os
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                        from app.utils.weather_api import WeatherAPIClient
                        
                        # Get state from session if available
                        selected_state = st.session_state.get(f"{key_prefix}state", "Punjab")
                        
                        weather_client = WeatherAPIClient()
                        weather_data = weather_client.get_current_weather(selected_state)
                        
                        # Store weather data in session state for auto-fill
                        st.session_state[f"{key_prefix}live_weather"] = weather_data
                        st.success(f"✅ Live weather data loaded for {selected_state}")
                        st.rerun()
                        
                    except Exception as e:
                        st.warning(f"⚠️ Could not fetch live weather data. Using manual input.")
        
        col1, col2 = st.columns(2)
        
        # Get live weather data if available
        live_weather = st.session_state.get(f"{key_prefix}live_weather", {})
        current_weather = live_weather.get('current', {})
        
        with col1:
            # Auto-fill from live weather if available
            default_temp = current_weather.get('temperature', 28.0)
            temperature = st.slider(
                "Temperature (°C)" + (" 🌡️ Live" if current_weather else ""),
                min_value=10.0,
                max_value=45.0,
                value=float(default_temp),
                step=0.5,
                key=f"{key_prefix}temp"
            )
            
            default_humidity = current_weather.get('humidity', 65.0)
            humidity = st.slider(
                "Humidity (%)" + (" 💧 Live" if current_weather else ""),
                min_value=30.0,
                max_value=95.0,
                value=float(default_humidity),
                step=1.0,
                key=f"{key_prefix}humidity"
            )
            
            # Rainfall - estimated based on weather condition
            default_rainfall = 2.5
            if current_weather.get('weather_condition') in ['Rainy', 'Heavy Rain', 'Thunderstorm']:
                default_rainfall = 15.0
            elif current_weather.get('weather_condition') in ['Drizzle', 'Light Rain']:
                default_rainfall = 5.0
            
            rainfall = st.slider(
                "Rainfall (mm)" + (" ☔ Estimated" if current_weather else ""),
                min_value=0.0,
                max_value=50.0,
                value=default_rainfall,
                step=0.1,
                key=f"{key_prefix}rainfall"
            )
        
        with col2:
            soil_ph = st.slider(
                "Soil pH",
                min_value=4.0,
                max_value=8.5,
                value=6.8,
                step=0.1,
                key=f"{key_prefix}soil_ph"
            )
            
            soil_organic_carbon = st.slider(
                "Soil Organic Carbon (%)",
                min_value=0.5,
                max_value=3.0,
                value=1.2,
                step=0.1,
                key=f"{key_prefix}soc"
            )
            
            soil_nitrogen = st.slider(
                "Soil Nitrogen (kg/ha)",
                min_value=100.0,
                max_value=500.0,
                value=280.0,
                step=10.0,
                key=f"{key_prefix}soil_n"
            )
        
        return {
            'temperature_celsius': temperature,
            'humidity_percent': humidity,
            'rainfall_mm': rainfall,
            'soil_ph': soil_ph,
            'soil_organic_carbon_percent': soil_organic_carbon,
            'soil_nitrogen_kg_per_ha': soil_nitrogen
        }
    
    @staticmethod
    def resource_input_form(key_prefix: str = "") -> Dict:
        """Create resource usage input form"""
        st.markdown("### 💧 Resource Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fertilizer_n = st.number_input(
                "Nitrogen Fertilizer (kg)",
                min_value=0.0,
                max_value=200.0,
                value=50.0,
                step=5.0,
                key=f"{key_prefix}fert_n"
            )
            
            fertilizer_p = st.number_input(
                "Phosphorus Fertilizer (kg)",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=2.5,
                key=f"{key_prefix}fert_p"
            )
        
        with col2:
            water_usage = st.number_input(
                "Water Usage (liters)",
                min_value=50.0,
                max_value=2000.0,
                value=200.0,
                step=10.0,
                key=f"{key_prefix}water"
            )
            
            pesticide_usage = st.number_input(
                "Pesticide Usage (kg)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key=f"{key_prefix}pesticide"
            )
        
        return {
            'fertilizer_n_kg': fertilizer_n,
            'fertilizer_p_kg': fertilizer_p,
            'water_usage_liters': water_usage,
            'pesticide_kg': pesticide_usage
        }

if __name__ == "__main__":
    # Demo of components (this would be used in the main dashboard)
    st.title("AgroMRV Dashboard Components Demo")
    
    # Header demo
    DashboardComponents.page_header(
        "AgroMRV System", 
        "Agricultural Monitoring, Reporting & Verification"
    )
    
    # Metrics demo
    sample_metrics = [
        {'label': 'Total Farms', 'value': 15, 'delta': '+2 this month', 'color': 'green'},
        {'label': 'Carbon Credits', 'value': 1245.67, 'delta': '+15.2%', 'color': 'blue'},
        {'label': 'Sustainability Score', 'value': 85.4, 'color': 'green'},
        {'label': 'Active States', 'value': 15, 'color': 'purple'}
    ]
    
    DashboardComponents.metrics_row(sample_metrics)
    
    # Status badges demo
    st.write("Status Examples:")
    DashboardComponents.status_badge('verified', 'Blockchain Verified')
    DashboardComponents.status_badge('active', 'AI Models')
    DashboardComponents.status_badge('pending', 'Processing')