"""
SmallholderMRVNode - Core MRV data generation for Indian agriculture
IPCC Tier 2 compliant agricultural data generation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmallholderMRVNode:
    """
    Core MRV (Monitoring, Reporting, Verification) data generator
    for Indian smallholder farms with IPCC Tier 2 compliance
    """
    
    # Indian states covered
    STATES = [
        'Punjab', 'Haryana', 'Uttar Pradesh', 'Bihar', 'West Bengal',
        'Madhya Pradesh', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Kerala',
        'Andhra Pradesh', 'Telangana', 'Gujarat', 'Rajasthan', 'Odisha'
    ]
    
    # Crop types with Indian focus
    CROPS = {
        'rice': {'season': 'kharif', 'carbon_factor': 0.42, 'n2o_factor': 0.013},
        'wheat': {'season': 'rabi', 'carbon_factor': 0.45, 'n2o_factor': 0.010},
        'maize': {'season': 'kharif', 'carbon_factor': 0.44, 'n2o_factor': 0.011},
        'vegetables': {'season': 'both', 'carbon_factor': 0.38, 'n2o_factor': 0.015},
        'millet': {'season': 'kharif', 'carbon_factor': 0.40, 'n2o_factor': 0.009},
        'sugarcane': {'season': 'annual', 'carbon_factor': 0.48, 'n2o_factor': 0.018},
        'cotton': {'season': 'kharif', 'carbon_factor': 0.41, 'n2o_factor': 0.012},
        'pulses': {'season': 'both', 'carbon_factor': 0.46, 'n2o_factor': 0.008}
    }
    
    def __init__(self, farm_id: str, state: str, crop_type: str, area_hectares: float):
        """
        Initialize MRV node for a specific farm
        
        Args:
            farm_id: Unique farm identifier
            state: Indian state name
            crop_type: Type of crop being grown
            area_hectares: Farm area in hectares
        """
        self.farm_id = farm_id
        self.state = state
        self.crop_type = crop_type
        self.area_hectares = area_hectares
        
        if state not in self.STATES:
            raise ValueError(f"State {state} not supported. Use one of: {self.STATES}")
        if crop_type not in self.CROPS:
            raise ValueError(f"Crop {crop_type} not supported. Use one of: {list(self.CROPS.keys())}")
            
        self.crop_info = self.CROPS[crop_type]
        self.created_at = datetime.now()
        
        logger.info(f"Initialized MRV Node: {farm_id} in {state} growing {crop_type}")
    
    def generate_historical_data(self, days: int = 60) -> pd.DataFrame:
        """
        Generate historical MRV data for specified number of days
        
        Args:
            days: Number of days of historical data
            
        Returns:
            DataFrame with complete MRV data
        """
        try:
            data = []
            start_date = datetime.now() - timedelta(days=days)
            
            for i in range(days):
                current_date = start_date + timedelta(days=i)
                daily_data = self._generate_daily_data(current_date)
                data.append(daily_data)
            
            df = pd.DataFrame(data)
            logger.info(f"Generated {days} days of historical data for farm {self.farm_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating historical data: {e}")
            raise
    
    def _generate_daily_data(self, date: datetime) -> Dict:
        """Generate comprehensive daily MRV data"""
        
        # Environmental factors with seasonal variations
        season_factor = self._get_seasonal_factor(date)
        
        # Weather data (realistic for Indian conditions)
        temperature = np.random.normal(
            28 + season_factor * 8,  # Base temp varies by season
            5
        )
        humidity = np.random.normal(65 + season_factor * 15, 10)
        rainfall = max(0, np.random.exponential(2.5) * season_factor)
        
        # Soil parameters
        soil_ph = np.random.normal(6.8, 0.5)
        soil_organic_carbon = np.random.normal(1.2, 0.3)
        soil_nitrogen = np.random.normal(280, 50)
        soil_phosphorus = np.random.normal(45, 15)
        
        # Resource usage
        water_usage = self._calculate_water_usage(temperature, humidity, rainfall)
        fertilizer_n = np.random.gamma(2, 15) * self.area_hectares
        fertilizer_p = np.random.gamma(2, 8) * self.area_hectares
        pesticide_usage = np.random.gamma(1.5, 3) * self.area_hectares
        
        # IPCC Tier 2 carbon calculations
        carbon_data = self._calculate_carbon_metrics(
            fertilizer_n, water_usage, temperature, soil_organic_carbon
        )
        
        # Yield and productivity
        yield_kg_per_ha = self._calculate_yield(temperature, rainfall, fertilizer_n)
        
        # Sustainability indicators
        sustainability = self._calculate_sustainability_indicators(
            soil_ph, soil_organic_carbon, pesticide_usage, water_usage
        )
        
        return {
            'date': date.strftime('%Y-%m-%d'),
            'farm_id': self.farm_id,
            'state': self.state,
            'crop_type': self.crop_type,
            'area_hectares': self.area_hectares,
            
            # Environmental
            'temperature_celsius': round(temperature, 2),
            'humidity_percent': round(max(30, min(95, humidity)), 2),
            'rainfall_mm': round(rainfall, 2),
            
            # Soil parameters
            'soil_ph': round(max(5.0, min(8.5, soil_ph)), 2),
            'soil_organic_carbon_percent': round(max(0.5, soil_organic_carbon), 3),
            'soil_nitrogen_kg_per_ha': round(max(100, soil_nitrogen), 2),
            'soil_phosphorus_kg_per_ha': round(max(10, soil_phosphorus), 2),
            
            # Resource usage
            'water_usage_liters': round(water_usage, 2),
            'fertilizer_n_kg': round(fertilizer_n, 2),
            'fertilizer_p_kg': round(fertilizer_p, 2),
            'pesticide_kg': round(pesticide_usage, 3),
            
            # Carbon and emissions (IPCC Tier 2)
            'co2_sequestered_kg': round(carbon_data['co2_sequestered'], 3),
            'co2_emissions_kg': round(carbon_data['co2_emissions'], 3),
            'n2o_emissions_kg': round(carbon_data['n2o_emissions'], 3),
            'ch4_emissions_kg': round(carbon_data['ch4_emissions'], 3),
            'net_carbon_balance_kg': round(carbon_data['net_carbon'], 3),
            
            # Productivity
            'yield_kg_per_ha': round(yield_kg_per_ha, 2),
            'biomass_kg_per_ha': round(yield_kg_per_ha * 1.6, 2),
            
            # Sustainability
            'sustainability_score': round(sustainability['overall'], 2),
            'soil_health_index': round(sustainability['soil_health'], 2),
            'water_efficiency': round(sustainability['water_efficiency'], 2),
            'biodiversity_index': round(sustainability['biodiversity'], 2),
            
            # Carbon credits (potential)
            'carbon_credits_potential': round(max(0, carbon_data['net_carbon'] * 0.0036), 4)
        }
    
    def _get_seasonal_factor(self, date: datetime) -> float:
        """Calculate seasonal variation factor (-1 to 1)"""
        day_of_year = date.timetuple().tm_yday
        # Peak summer around day 150, winter around day 350
        return np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    def _calculate_water_usage(self, temp: float, humidity: float, rainfall: float) -> float:
        """Calculate daily water usage based on environmental conditions"""
        base_usage = 50 * self.area_hectares  # Base 50 L/ha
        
        # Temperature effect
        temp_factor = max(0.5, (temp - 20) / 15)
        
        # Humidity effect (inverse)
        humidity_factor = max(0.3, (100 - humidity) / 80)
        
        # Rainfall reduction
        rainfall_reduction = min(0.8, rainfall / 20)
        
        return base_usage * temp_factor * humidity_factor * (1 - rainfall_reduction)
    
    def _calculate_carbon_metrics(self, fertilizer_n: float, water_usage: float, 
                                 temp: float, soc: float) -> Dict[str, float]:
        """Calculate IPCC Tier 2 compliant carbon metrics"""
        
        # CO2 sequestration (enhanced by organic practices)
        base_sequestration = self.area_hectares * self.crop_info['carbon_factor'] * 1000
        soc_bonus = soc * 0.2  # Bonus for higher soil organic carbon
        co2_sequestered = base_sequestration * (1 + soc_bonus)
        
        # CO2 emissions from inputs
        fertilizer_emissions = fertilizer_n * 2.6  # kg CO2 per kg N
        machinery_emissions = self.area_hectares * 15  # Daily machinery use
        co2_emissions = fertilizer_emissions + machinery_emissions
        
        # N2O emissions from fertilizer (IPCC default)
        n2o_direct = fertilizer_n * self.crop_info['n2o_factor']
        n2o_indirect = fertilizer_n * 0.0075  # Indirect emissions
        n2o_total = (n2o_direct + n2o_indirect) * 298  # GWP conversion
        
        # CH4 emissions (mainly for rice)
        if self.crop_type == 'rice':
            ch4_emissions = self.area_hectares * 20 * (1 + temp / 100)  # Temperature dependent
        else:
            ch4_emissions = self.area_hectares * 2
            
        ch4_total = ch4_emissions * 25  # GWP conversion
        
        net_carbon = co2_sequestered - co2_emissions - n2o_total - ch4_total
        
        return {
            'co2_sequestered': co2_sequestered,
            'co2_emissions': co2_emissions,
            'n2o_emissions': n2o_total,
            'ch4_emissions': ch4_total,
            'net_carbon': net_carbon
        }
    
    def _calculate_yield(self, temp: float, rainfall: float, fertilizer: float) -> float:
        """Calculate crop yield based on environmental factors"""
        
        # Base yield by crop type (kg/ha)
        base_yields = {
            'rice': 4000, 'wheat': 3200, 'maize': 3800, 'vegetables': 15000,
            'millet': 1800, 'sugarcane': 70000, 'cotton': 500, 'pulses': 1200
        }
        
        base_yield = base_yields.get(self.crop_type, 2000)
        
        # Temperature optimization (crop specific)
        optimal_temps = {
            'rice': 28, 'wheat': 22, 'maize': 30, 'vegetables': 25,
            'millet': 32, 'sugarcane': 30, 'cotton': 28, 'pulses': 25
        }
        
        optimal_temp = optimal_temps.get(self.crop_type, 25)
        temp_factor = 1 - abs(temp - optimal_temp) / 20
        temp_factor = max(0.3, min(1.2, temp_factor))
        
        # Rainfall factor (minimum threshold needed)
        rain_factor = min(1.5, rainfall / 5 + 0.5)
        
        # Fertilizer response (diminishing returns)
        fert_factor = 1 + (fertilizer / 100) / (1 + fertilizer / 100)
        
        return base_yield * temp_factor * rain_factor * fert_factor
    
    def _calculate_sustainability_indicators(self, ph: float, soc: float, 
                                           pesticide: float, water: float) -> Dict[str, float]:
        """Calculate sustainability indicators"""
        
        # Soil health (pH and organic carbon)
        ph_score = 1 - abs(ph - 6.8) / 2  # Optimal around 6.8
        soc_score = min(1, soc / 2)  # Higher is better
        soil_health = (ph_score + soc_score) * 50
        
        # Water efficiency (less usage is better for same yield)
        water_efficiency = max(20, 100 - water / (self.area_hectares * 100))
        
        # Biodiversity (inverse of pesticide usage)
        biodiversity = max(30, 100 - pesticide * 10)
        
        # Overall sustainability
        overall = (soil_health + water_efficiency + biodiversity) / 3
        
        return {
            'soil_health': soil_health,
            'water_efficiency': water_efficiency,
            'biodiversity': biodiversity,
            'overall': overall
        }
    
    def get_farm_summary(self) -> Dict:
        """Get comprehensive farm summary"""
        return {
            'farm_id': self.farm_id,
            'state': self.state,
            'crop_type': self.crop_type,
            'area_hectares': self.area_hectares,
            'season': self.crop_info['season'],
            'created_at': self.created_at.isoformat(),
            'mrv_compliance': 'IPCC Tier 2',
            'supported_metrics': [
                'Carbon Sequestration', 'GHG Emissions', 'Yield Prediction',
                'Water Usage', 'Soil Health', 'Sustainability Scoring'
            ]
        }

def create_demo_farms() -> List[SmallholderMRVNode]:
    """Create representative demo farms across India"""
    demo_configs = [
        ('FARM001', 'Punjab', 'wheat', 2.5),
        ('FARM002', 'Tamil Nadu', 'rice', 1.8),
        ('FARM003', 'Maharashtra', 'cotton', 3.2),
        ('FARM004', 'Uttar Pradesh', 'sugarcane', 4.0),
        ('FARM005', 'Haryana', 'maize', 2.0),
        ('FARM006', 'Karnataka', 'vegetables', 1.2),
        ('FARM007', 'West Bengal', 'rice', 2.8),
        ('FARM008', 'Madhya Pradesh', 'pulses', 3.5),
        ('FARM009', 'Gujarat', 'cotton', 2.5),
        ('FARM010', 'Bihar', 'wheat', 1.5),
        ('FARM011', 'Rajasthan', 'millet', 4.5),
        ('FARM012', 'Andhra Pradesh', 'rice', 2.2),
        ('FARM013', 'Kerala', 'vegetables', 0.8),
        ('FARM014', 'Telangana', 'maize', 3.0),
        ('FARM015', 'Odisha', 'rice', 1.9)
    ]
    
    farms = []
    for config in demo_configs:
        try:
            farm = SmallholderMRVNode(*config)
            farms.append(farm)
        except Exception as e:
            logger.error(f"Failed to create farm {config[0]}: {e}")
    
    logger.info(f"Created {len(farms)} demo farms")
    return farms

if __name__ == "__main__":
    # Demo usage
    farm = SmallholderMRVNode('DEMO001', 'Punjab', 'wheat', 2.5)
    data = farm.generate_historical_data(30)
    print("Sample MRV Data:")
    print(data.head())
    print(f"\nTotal records: {len(data)}")
    print(f"Average carbon balance: {data['net_carbon_balance_kg'].mean():.2f} kg")