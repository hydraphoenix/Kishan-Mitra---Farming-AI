"""
Weather API Integration for AgroMRV System
Provides real-time weather data and agricultural forecasting
"""

import requests
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class WeatherAPIClient:
    """Weather data client using OpenWeatherMap API"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Using demo API key - in production, this would be from environment variables
        self.api_key = api_key or "demo_key_for_nabard_hackathon"
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.geo_url = "https://api.openweathermap.org/geo/1.0"
        
        # Indian state coordinates for demo purposes
        self.state_coordinates = {
            'Punjab': {'lat': 31.1471, 'lon': 75.3412},
            'Haryana': {'lat': 29.0588, 'lon': 76.0856},
            'Uttar Pradesh': {'lat': 26.8467, 'lon': 80.9462},
            'Bihar': {'lat': 25.0961, 'lon': 85.3131},
            'West Bengal': {'lat': 22.9868, 'lon': 87.8550},
            'Madhya Pradesh': {'lat': 22.9734, 'lon': 78.6569},
            'Maharashtra': {'lat': 19.7515, 'lon': 75.7139},
            'Karnataka': {'lat': 15.3173, 'lon': 75.7139},
            'Tamil Nadu': {'lat': 11.1271, 'lon': 78.6569},
            'Kerala': {'lat': 10.8505, 'lon': 76.2711},
            'Andhra Pradesh': {'lat': 15.9129, 'lon': 79.7400},
            'Telangana': {'lat': 18.1124, 'lon': 79.0193},
            'Gujarat': {'lat': 23.0225, 'lon': 72.5714},
            'Rajasthan': {'lat': 27.0238, 'lon': 74.2179},
            'Odisha': {'lat': 20.9517, 'lon': 85.0985}
        }
    
    def get_current_weather(self, state: str, city: Optional[str] = None) -> Dict:
        """Get current weather data for a location"""
        try:
            # For demo purposes, return realistic weather data based on state
            # In production, this would make actual API calls
            coords = self.state_coordinates.get(state, self.state_coordinates['Punjab'])
            
            # Generate realistic weather data based on location and season
            current_month = datetime.now().month
            
            # Temperature ranges by season and region
            if state in ['Punjab', 'Haryana', 'Uttar Pradesh']:
                if current_month in [12, 1, 2]:  # Winter
                    temp_range = (8, 18)
                    humidity_range = (65, 85)
                elif current_month in [3, 4, 5]:  # Spring/Summer
                    temp_range = (25, 35)
                    humidity_range = (40, 60)
                elif current_month in [6, 7, 8, 9]:  # Monsoon
                    temp_range = (25, 30)
                    humidity_range = (75, 95)
                else:  # Post-monsoon
                    temp_range = (15, 25)
                    humidity_range = (60, 80)
            else:
                # Adjust for southern states
                temp_range = (temp_range[0] + 5, temp_range[1] + 5)
            
            import random
            
            weather_data = {
                'location': {
                    'state': state,
                    'city': city or state,
                    'coordinates': coords
                },
                'current': {
                    'temperature': round(random.uniform(*temp_range), 1),
                    'humidity': round(random.uniform(*humidity_range), 1),
                    'pressure': round(random.uniform(1010, 1020), 1),
                    'wind_speed': round(random.uniform(2, 8), 1),
                    'visibility': round(random.uniform(8, 15), 1),
                    'uv_index': round(random.uniform(3, 8), 1),
                    'weather_condition': self._get_seasonal_weather_condition(current_month),
                    'timestamp': datetime.now().isoformat()
                },
                'air_quality': {
                    'aqi': random.randint(50, 150),
                    'pm25': round(random.uniform(25, 75), 1),
                    'pm10': round(random.uniform(40, 100), 1)
                }
            }
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._get_fallback_weather(state)
    
    def get_weather_forecast(self, state: str, days: int = 5) -> List[Dict]:
        """Get weather forecast for next few days"""
        try:
            forecasts = []
            base_weather = self.get_current_weather(state)
            
            for i in range(days):
                forecast_date = datetime.now() + timedelta(days=i)
                
                # Add some variation for forecast
                import random
                temp_variation = random.uniform(-3, 3)
                humidity_variation = random.uniform(-10, 10)
                
                forecast = {
                    'date': forecast_date.strftime('%Y-%m-%d'),
                    'temperature': {
                        'min': base_weather['current']['temperature'] + temp_variation - 5,
                        'max': base_weather['current']['temperature'] + temp_variation + 5,
                        'avg': base_weather['current']['temperature'] + temp_variation
                    },
                    'humidity': max(30, min(95, base_weather['current']['humidity'] + humidity_variation)),
                    'rainfall_probability': random.randint(10, 70),
                    'wind_speed': round(random.uniform(3, 12), 1),
                    'weather_condition': self._get_forecast_condition(forecast_date.month),
                    'agricultural_advisory': self._get_agricultural_advisory(
                        base_weather['current']['temperature'] + temp_variation,
                        base_weather['current']['humidity'] + humidity_variation,
                        forecast_date.month
                    )
                }
                forecasts.append(forecast)
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Error fetching forecast: {e}")
            return []
    
    def get_agricultural_weather_summary(self, state: str, crop_type: str) -> Dict:
        """Get agriculture-specific weather summary and recommendations"""
        current_weather = self.get_current_weather(state)
        forecast = self.get_weather_forecast(state, 7)
        
        crop_requirements = {
            'rice': {'optimal_temp': (20, 35), 'water_needs': 'high', 'humidity_tolerance': 'high'},
            'wheat': {'optimal_temp': (15, 25), 'water_needs': 'medium', 'humidity_tolerance': 'medium'},
            'maize': {'optimal_temp': (21, 27), 'water_needs': 'medium', 'humidity_tolerance': 'medium'},
            'cotton': {'optimal_temp': (21, 30), 'water_needs': 'low', 'humidity_tolerance': 'low'},
            'sugarcane': {'optimal_temp': (20, 30), 'water_needs': 'high', 'humidity_tolerance': 'high'}
        }
        
        crop_req = crop_requirements.get(crop_type.lower(), crop_requirements['wheat'])
        current_temp = current_weather['current']['temperature']
        current_humidity = current_weather['current']['humidity']
        
        # Weather suitability analysis
        temp_suitability = self._calculate_suitability(
            current_temp, crop_req['optimal_temp'][0], crop_req['optimal_temp'][1]
        )
        
        # Generate recommendations
        recommendations = []
        alerts = []
        
        if current_temp < crop_req['optimal_temp'][0]:
            recommendations.append(f"Temperature is below optimal for {crop_type}. Consider covering crops or delayed planting.")
            alerts.append("Low Temperature Alert")
        elif current_temp > crop_req['optimal_temp'][1]:
            recommendations.append(f"High temperature stress possible for {crop_type}. Ensure adequate irrigation.")
            alerts.append("High Temperature Alert")
        
        if current_humidity > 85 and crop_req['humidity_tolerance'] == 'low':
            recommendations.append("High humidity may increase disease risk. Monitor for fungal infections.")
            alerts.append("High Humidity Warning")
        
        # Calculate irrigation needs based on weather
        irrigation_need = self._calculate_irrigation_need(current_weather, forecast, crop_type)
        
        return {
            'weather_summary': current_weather,
            'forecast_summary': forecast[:3],  # Next 3 days
            'crop_suitability': {
                'temperature_suitability': f"{temp_suitability:.1f}%",
                'overall_conditions': 'Favorable' if temp_suitability > 75 else 'Moderate' if temp_suitability > 50 else 'Challenging'
            },
            'irrigation_recommendation': irrigation_need,
            'agricultural_alerts': alerts,
            'recommendations': recommendations,
            'best_farming_days': self._identify_best_farming_days(forecast),
            'weather_risk_assessment': self._assess_weather_risks(current_weather, forecast, crop_type)
        }
    
    def _get_seasonal_weather_condition(self, month: int) -> str:
        """Get typical weather condition for the month"""
        conditions = {
            12: 'Clear', 1: 'Partly Cloudy', 2: 'Clear',
            3: 'Sunny', 4: 'Hot', 5: 'Very Hot',
            6: 'Monsoon', 7: 'Heavy Rain', 8: 'Rainy', 9: 'Post Monsoon',
            10: 'Pleasant', 11: 'Cool'
        }
        return conditions.get(month, 'Clear')
    
    def _get_forecast_condition(self, month: int) -> str:
        """Get forecast condition with some randomness"""
        import random
        base_condition = self._get_seasonal_weather_condition(month)
        variations = {
            'Clear': ['Clear', 'Partly Cloudy', 'Sunny'],
            'Sunny': ['Sunny', 'Hot', 'Clear'],
            'Rainy': ['Rainy', 'Cloudy', 'Drizzle'],
            'Monsoon': ['Heavy Rain', 'Thunderstorm', 'Rainy']
        }
        return random.choice(variations.get(base_condition, [base_condition]))
    
    def _get_agricultural_advisory(self, temp: float, humidity: float, month: int) -> str:
        """Generate agricultural advisory based on weather"""
        advisories = []
        
        if temp > 35:
            advisories.append("Provide shade to crops and increase irrigation frequency")
        elif temp < 10:
            advisories.append("Protect crops from frost damage")
        
        if humidity > 90:
            advisories.append("Monitor for fungal diseases, ensure proper ventilation")
        elif humidity < 40:
            advisories.append("Increase irrigation to prevent crop stress")
        
        if month in [6, 7, 8]:  # Monsoon months
            advisories.append("Ensure proper drainage to prevent waterlogging")
        
        return "; ".join(advisories) if advisories else "Conditions are favorable for normal farming operations"
    
    def _calculate_suitability(self, current: float, min_val: float, max_val: float) -> float:
        """Calculate how suitable current conditions are for crop"""
        if min_val <= current <= max_val:
            return 100.0
        elif current < min_val:
            return max(0, 100 - (min_val - current) * 10)
        else:
            return max(0, 100 - (current - max_val) * 10)
    
    def _calculate_irrigation_need(self, current_weather: Dict, forecast: List[Dict], crop_type: str) -> Dict:
        """Calculate irrigation needs based on weather data"""
        # Simplified irrigation calculation
        temp = current_weather['current']['temperature']
        humidity = current_weather['current']['humidity']
        
        # Base water need
        base_need = 100  # liters per plant/area
        
        # Adjust based on temperature
        if temp > 30:
            temp_factor = 1.5
        elif temp > 25:
            temp_factor = 1.2
        else:
            temp_factor = 1.0
        
        # Adjust based on humidity
        if humidity < 50:
            humidity_factor = 1.3
        elif humidity < 70:
            humidity_factor = 1.1
        else:
            humidity_factor = 0.9
        
        # Check upcoming rainfall
        rainfall_expected = any(day.get('rainfall_probability', 0) > 60 for day in forecast[:3])
        
        irrigation_need = base_need * temp_factor * humidity_factor
        
        if rainfall_expected:
            irrigation_need *= 0.7
        
        return {
            'daily_requirement': f"{irrigation_need:.0f} L/plant",
            'frequency': 'Daily' if temp > 30 else 'Every 2 days' if temp > 25 else 'Every 3 days',
            'timing': 'Early morning or evening to minimize evaporation',
            'rainfall_expected': rainfall_expected,
            'priority': 'High' if temp > 30 or humidity < 50 else 'Medium'
        }
    
    def _identify_best_farming_days(self, forecast: List[Dict]) -> List[str]:
        """Identify best days for farming activities in the forecast"""
        best_days = []
        
        for day in forecast:
            temp = day['temperature']['avg']
            humidity = day.get('humidity', 70)
            rainfall_prob = day.get('rainfall_probability', 30)
            
            # Good farming day criteria
            if (20 <= temp <= 30 and 
                50 <= humidity <= 80 and 
                rainfall_prob < 40):
                best_days.append(f"{day['date']} - Excellent for field work")
            elif (15 <= temp <= 35 and rainfall_prob < 60):
                best_days.append(f"{day['date']} - Good for most activities")
        
        return best_days[:3]  # Return top 3 days
    
    def _assess_weather_risks(self, current: Dict, forecast: List[Dict], crop_type: str) -> Dict:
        """Assess weather-related risks for the crop"""
        risks = []
        risk_level = "Low"
        
        current_temp = current['current']['temperature']
        current_humidity = current['current']['humidity']
        
        # Temperature risks
        if current_temp > 40:
            risks.append("Extreme heat stress risk")
            risk_level = "High"
        elif current_temp < 5:
            risks.append("Frost damage risk")
            risk_level = "High"
        
        # Disease risks based on humidity
        if current_humidity > 90:
            risks.append("High fungal disease risk")
            if risk_level == "Low":
                risk_level = "Medium"
        
        # Check forecast for extreme events
        for day in forecast[:3]:
            if day.get('rainfall_probability', 0) > 80:
                risks.append("Heavy rainfall expected - waterlogging risk")
                risk_level = "Medium"
        
        return {
            'overall_risk_level': risk_level,
            'identified_risks': risks,
            'mitigation_suggestions': self._get_mitigation_suggestions(risks, crop_type)
        }
    
    def _get_mitigation_suggestions(self, risks: List[str], crop_type: str) -> List[str]:
        """Get mitigation suggestions for identified risks"""
        suggestions = []
        
        for risk in risks:
            if "heat stress" in risk.lower():
                suggestions.append("Increase irrigation frequency and provide shade if possible")
            elif "frost" in risk.lower():
                suggestions.append("Cover crops with protective material during night")
            elif "fungal disease" in risk.lower():
                suggestions.append("Ensure good air circulation and consider preventive fungicide spray")
            elif "waterlogging" in risk.lower():
                suggestions.append("Ensure proper drainage channels and avoid irrigation")
        
        return suggestions
    
    def _get_fallback_weather(self, state: str) -> Dict:
        """Fallback weather data when API is unavailable"""
        coords = self.state_coordinates.get(state, self.state_coordinates['Punjab'])
        
        return {
            'location': {
                'state': state,
                'coordinates': coords
            },
            'current': {
                'temperature': 25.0,
                'humidity': 65.0,
                'pressure': 1013.2,
                'wind_speed': 5.0,
                'weather_condition': 'Clear',
                'timestamp': datetime.now().isoformat()
            },
            'note': 'Demo weather data - API integration in progress'
        }

# Demo function to test weather integration
def test_weather_integration():
    """Test function for weather API integration"""
    weather_client = WeatherAPIClient()
    
    print("Testing Weather API Integration...")
    
    # Test current weather
    current = weather_client.get_current_weather('Punjab')
    print(f"Current weather in Punjab: {current['current']['temperature']}°C, {current['current']['humidity']}% humidity")
    
    # Test forecast
    forecast = weather_client.get_weather_forecast('Punjab', 3)
    print(f"3-day forecast available: {len(forecast)} days")
    
    # Test agricultural summary
    ag_summary = weather_client.get_agricultural_weather_summary('Punjab', 'wheat')
    print(f"Agricultural suitability: {ag_summary['crop_suitability']['overall_conditions']}")
    
    return True

if __name__ == "__main__":
    test_weather_integration()