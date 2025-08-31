"""
Advanced Farm Recommendations Engine
Data-driven actionable insights for yield optimization, profitability, and sustainability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FarmRecommendationsEngine:
    """Comprehensive farm optimization recommendations"""
    
    def __init__(self):
        self.crop_data = self._initialize_crop_database()
        self.regional_data = self._initialize_regional_database()
        self.market_prices = self._initialize_market_data()
        
    def _initialize_crop_database(self) -> Dict:
        """Initialize crop-specific optimization parameters"""
        return {
            'rice': {
                'optimal_temp_range': (20, 35),
                'optimal_ph_range': (5.5, 6.5),
                'optimal_water_depth': 5,  # cm
                'fertilizer_n_optimal': 120,  # kg/ha
                'fertilizer_p_optimal': 60,   # kg/ha
                'fertilizer_k_optimal': 40,   # kg/ha
                'planting_density': 25,       # plants/m²
                'growth_duration': 120,       # days
                'water_requirement': 1500,    # mm/season
                'yield_potential': 6000,      # kg/ha
                'common_pests': ['brown_planthopper', 'stem_borer', 'leaf_folder'],
                'disease_risks': ['blast', 'bacterial_blight', 'sheath_blight'],
                'mechanization_priority': ['transplanter', 'combine_harvester']
            },
            'wheat': {
                'optimal_temp_range': (15, 25),
                'optimal_ph_range': (6.0, 7.5),
                'optimal_water_depth': 0,
                'fertilizer_n_optimal': 150,
                'fertilizer_p_optimal': 80,
                'fertilizer_k_optimal': 60,
                'planting_density': 100,
                'growth_duration': 120,
                'water_requirement': 450,
                'yield_potential': 4500,
                'common_pests': ['aphids', 'termites', 'armyworm'],
                'disease_risks': ['rust', 'smut', 'bunt'],
                'mechanization_priority': ['seed_drill', 'combine_harvester']
            },
            'maize': {
                'optimal_temp_range': (21, 27),
                'optimal_ph_range': (6.0, 6.8),
                'optimal_water_depth': 0,
                'fertilizer_n_optimal': 200,
                'fertilizer_p_optimal': 80,
                'fertilizer_k_optimal': 80,
                'planting_density': 75,
                'growth_duration': 100,
                'water_requirement': 500,
                'yield_potential': 8000,
                'common_pests': ['fall_armyworm', 'stem_borer', 'cutworm'],
                'disease_risks': ['blight', 'rust', 'stalk_rot'],
                'mechanization_priority': ['planter', 'cultivator']
            },
            'cotton': {
                'optimal_temp_range': (21, 30),
                'optimal_ph_range': (5.8, 8.0),
                'optimal_water_depth': 0,
                'fertilizer_n_optimal': 160,
                'fertilizer_p_optimal': 80,
                'fertilizer_k_optimal': 80,
                'planting_density': 12,
                'growth_duration': 180,
                'water_requirement': 800,
                'yield_potential': 2500,
                'common_pests': ['bollworm', 'aphids', 'whitefly'],
                'disease_risks': ['wilt', 'blight', 'leaf_curl'],
                'mechanization_priority': ['cotton_picker', 'cultivator']
            }
        }
    
    def _initialize_regional_database(self) -> Dict:
        """Initialize region-specific recommendations"""
        return {
            'Punjab': {
                'climate_zone': 'subtropical',
                'major_challenges': ['water_scarcity', 'soil_salinity', 'pest_resistance'],
                'recommended_practices': ['laser_land_leveling', 'drip_irrigation', 'integrated_pest_management'],
                'government_schemes': ['PM-KISAN', 'soil_health_card', 'pradhan_mantri_fasal_bima_yojana'],
                'market_access': 'excellent',
                'avg_farm_size': 3.62
            },
            'Haryana': {
                'climate_zone': 'subtropical',
                'major_challenges': ['groundwater_depletion', 'crop_residue_burning'],
                'recommended_practices': ['crop_diversification', 'happy_seeder', 'micro_irrigation'],
                'government_schemes': ['mera_paani_meri_virasat', 'crop_residue_management'],
                'market_access': 'good',
                'avg_farm_size': 2.28
            },
            'Uttar Pradesh': {
                'climate_zone': 'subtropical',
                'major_challenges': ['fragmented_holdings', 'low_mechanization'],
                'recommended_practices': ['custom_hiring_centers', 'farmer_producer_organizations'],
                'government_schemes': ['kisan_credit_card', 'pm_kisan_sampada_yojana'],
                'market_access': 'moderate',
                'avg_farm_size': 0.79
            }
        }
    
    def _initialize_market_data(self) -> Dict:
        """Initialize market price and demand data"""
        return {
            'rice': {'msp': 2183, 'market_avg': 2250, 'demand_trend': 'stable', 'export_potential': 'high'},
            'wheat': {'msp': 2275, 'market_avg': 2320, 'demand_trend': 'stable', 'export_potential': 'medium'},
            'maize': {'msp': 1962, 'market_avg': 2100, 'demand_trend': 'growing', 'export_potential': 'medium'},
            'cotton': {'msp': 6080, 'market_avg': 6200, 'demand_trend': 'volatile', 'export_potential': 'high'},
        }
    
    def generate_comprehensive_recommendations(self, farm_data: Dict) -> Dict:
        """Generate comprehensive actionable recommendations"""
        
        crop_type = farm_data.get('crop_type', 'wheat').lower()
        state = farm_data.get('state', 'Punjab')
        area_hectares = float(farm_data.get('area_hectares', 2.5))
        
        recommendations = {
            'crop_management': self._get_crop_management_recommendations(farm_data),
            'resource_optimization': self._get_resource_optimization_recommendations(farm_data),
            'soil_health': self._get_soil_health_recommendations(farm_data),
            'mechanization': self._get_mechanization_recommendations(farm_data),
            'cost_optimization': self._get_cost_optimization_recommendations(farm_data),
            'sustainability': self._get_sustainability_recommendations(farm_data),
            'market_strategy': self._get_market_strategy_recommendations(farm_data),
            'priority_actions': self._get_priority_actions(farm_data),
            'roi_projections': self._calculate_roi_projections(farm_data)
        }
        
        return recommendations
    
    def _get_crop_management_recommendations(self, farm_data: Dict) -> List[Dict]:
        """Generate crop management practice recommendations"""
        crop_type = farm_data.get('crop_type', 'wheat').lower()
        current_yield = farm_data.get('yield_kg_per_ha', 3000)
        area = float(farm_data.get('area_hectares', 2.5))
        
        crop_info = self.crop_data.get(crop_type, self.crop_data['wheat'])
        potential_yield = crop_info['yield_potential']
        yield_gap = potential_yield - current_yield
        
        recommendations = []
        
        # Yield gap analysis
        if yield_gap > 1000:
            recommendations.append({
                'category': 'Variety Selection',
                'priority': 'High',
                'recommendation': f'Switch to high-yielding varieties (HYV) to bridge {yield_gap:.0f} kg/ha yield gap',
                'expected_benefit': f'+{yield_gap * 0.6:.0f} kg/ha yield increase',
                'implementation_cost': f'₹{area * 3000:.0f}',
                'payback_period': '1 season',
                'action_items': [
                    f'Source certified {crop_type} seeds from authorized dealers',
                    'Conduct soil test before variety selection',
                    'Train on new variety-specific management practices'
                ]
            })
        
        # Planting optimization
        current_density = farm_data.get('planting_density', crop_info['planting_density'] * 0.8)
        optimal_density = crop_info['planting_density']
        
        if abs(current_density - optimal_density) > optimal_density * 0.2:
            recommendations.append({
                'category': 'Planting Density',
                'priority': 'Medium',
                'recommendation': f'Optimize planting density to {optimal_density} plants/m² for maximum yield',
                'expected_benefit': '+8-12% yield improvement',
                'implementation_cost': f'₹{area * 500:.0f}',
                'payback_period': '1 season',
                'action_items': [
                    'Use seed drill for precise spacing',
                    'Calculate seed rate based on germination percentage',
                    'Monitor and thin overcrowded areas'
                ]
            })
        
        # Growth stage management
        recommendations.append({
            'category': 'Growth Stage Management',
            'priority': 'High',
            'recommendation': 'Implement stage-specific nutrient and water management',
            'expected_benefit': '+15-20% resource use efficiency',
            'implementation_cost': f'₹{area * 1000:.0f}',
            'payback_period': '1 season',
            'action_items': [
                'Create crop calendar with critical growth stages',
                'Schedule inputs based on growth requirements',
                'Monitor crop health weekly using mobile apps'
            ]
        })
        
        return recommendations
    
    def _get_resource_optimization_recommendations(self, farm_data: Dict) -> List[Dict]:
        """Generate resource allocation and optimization recommendations"""
        crop_type = farm_data.get('crop_type', 'wheat').lower()
        current_n = farm_data.get('fertilizer_n_kg', 100)
        current_water = farm_data.get('water_usage_liters', 200000)
        area = float(farm_data.get('area_hectares', 2.5))
        
        crop_info = self.crop_data.get(crop_type, self.crop_data['wheat'])
        recommendations = []
        
        # Fertilizer optimization
        optimal_n = crop_info['fertilizer_n_optimal']
        n_efficiency = min(current_n / optimal_n, 1.0) if optimal_n > 0 else 0.8
        
        if abs(current_n - optimal_n) > optimal_n * 0.2:
            recommendations.append({
                'category': 'Nutrient Management',
                'priority': 'High',
                'recommendation': f'Optimize nitrogen application to {optimal_n} kg/ha with split application',
                'expected_benefit': f'+₹{area * 5000:.0f} additional profit through better nutrient efficiency',
                'implementation_cost': f'₹{area * 2000:.0f}',
                'payback_period': '1 season',
                'action_items': [
                    'Conduct soil nutrient analysis',
                    'Apply 50% N at planting, 30% at tillering, 20% at flowering',
                    'Use slow-release fertilizers or neem-coated urea',
                    'Monitor plant tissue for nutrient status'
                ]
            })
        
        # Water optimization
        optimal_water = crop_info['water_requirement'] * area * 10000  # Convert mm to liters
        if current_water > optimal_water * 1.3:
            water_savings = current_water - optimal_water
            recommendations.append({
                'category': 'Water Management',
                'priority': 'High',
                'recommendation': f'Reduce water usage by {water_savings/10000:.0f}% through precision irrigation',
                'expected_benefit': f'Save ₹{water_savings * 0.02:.0f} in water costs + 10% yield improvement',
                'implementation_cost': f'₹{area * 15000:.0f} (drip system)',
                'payback_period': '2-3 seasons',
                'action_items': [
                    'Install soil moisture sensors',
                    'Implement drip or sprinkler irrigation',
                    'Schedule irrigation based on crop water requirements',
                    'Use mulching to reduce evaporation losses'
                ]
            })
        
        # Integrated pest management
        current_pesticide = farm_data.get('pesticide_kg', 5)
        if current_pesticide > 3:
            recommendations.append({
                'category': 'Pest Management',
                'priority': 'Medium',
                'recommendation': 'Implement Integrated Pest Management (IPM) to reduce chemical pesticide use',
                'expected_benefit': f'Reduce pesticide costs by 40% (₹{area * 2000:.0f})',
                'implementation_cost': f'₹{area * 1000:.0f}',
                'payback_period': '1 season',
                'action_items': [
                    'Use pheromone traps for pest monitoring',
                    'Introduce beneficial insects (biocontrol agents)',
                    'Apply neem-based organic pesticides',
                    'Rotate crops to break pest cycles'
                ]
            })
        
        return recommendations
    
    def _get_soil_health_recommendations(self, farm_data: Dict) -> List[Dict]:
        """Generate soil health improvement recommendations"""
        soil_ph = farm_data.get('soil_ph', 6.5)
        soil_organic_matter = farm_data.get('soil_organic_matter_percent', 2.0)
        area = float(farm_data.get('area_hectares', 2.5))
        crop_type = farm_data.get('crop_type', 'wheat').lower()
        
        crop_info = self.crop_data.get(crop_type, self.crop_data['wheat'])
        optimal_ph_range = crop_info['optimal_ph_range']
        
        recommendations = []
        
        # pH correction
        if soil_ph < optimal_ph_range[0] - 0.2:
            lime_requirement = (optimal_ph_range[0] - soil_ph) * area * 500  # kg
            recommendations.append({
                'category': 'Soil pH Management',
                'priority': 'High',
                'recommendation': f'Apply agricultural lime to increase soil pH from {soil_ph} to {optimal_ph_range[0]}',
                'expected_benefit': '+15-20% nutrient availability improvement',
                'implementation_cost': f'₹{lime_requirement * 8:.0f}',
                'payback_period': '2-3 seasons',
                'action_items': [
                    f'Apply {lime_requirement:.0f} kg agricultural lime',
                    'Incorporate lime 2-3 weeks before planting',
                    'Retest soil pH after 6 months'
                ]
            })
        elif soil_ph > optimal_ph_range[1] + 0.3:
            recommendations.append({
                'category': 'Soil pH Management',
                'priority': 'High',
                'recommendation': f'Apply gypsum or sulfur to reduce soil pH from {soil_ph}',
                'expected_benefit': 'Improved nutrient uptake and reduced alkalinity stress',
                'implementation_cost': f'₹{area * 3000:.0f}',
                'payback_period': '2 seasons',
                'action_items': [
                    f'Apply {area * 500:.0f} kg gypsum per hectare',
                    'Ensure adequate drainage',
                    'Add organic matter to buffer pH changes'
                ]
            })
        
        # Organic matter enhancement
        if soil_organic_matter < 2.5:
            recommendations.append({
                'category': 'Soil Organic Matter',
                'priority': 'Medium',
                'recommendation': f'Increase soil organic matter from {soil_organic_matter}% to 3.0%',
                'expected_benefit': 'Improved soil structure, water retention, and nutrient cycling',
                'implementation_cost': f'₹{area * 4000:.0f}',
                'payback_period': '3-4 seasons',
                'action_items': [
                    f'Apply {area * 5:.0f} tons farmyard manure annually',
                    'Practice crop residue incorporation',
                    'Grow green manure crops in fallow periods',
                    'Use compost and vermicompost'
                ]
            })
        
        # Soil conservation
        recommendations.append({
            'category': 'Soil Conservation',
            'priority': 'Medium',
            'recommendation': 'Implement soil conservation practices to prevent erosion',
            'expected_benefit': 'Preserve topsoil and maintain long-term productivity',
            'implementation_cost': f'₹{area * 2000:.0f}',
            'payback_period': 'Long-term',
            'action_items': [
                'Practice contour farming on slopes',
                'Establish bunds and terraces',
                'Maintain permanent vegetation strips',
                'Use cover crops during fallow periods'
            ]
        })
        
        return recommendations
    
    def _get_mechanization_recommendations(self, farm_data: Dict) -> List[Dict]:
        """Generate mechanization and technology adoption recommendations"""
        area = float(farm_data.get('area_hectares', 2.5))
        mechanization_level = farm_data.get('mechanization_level', 'medium')
        crop_type = farm_data.get('crop_type', 'wheat').lower()
        
        crop_info = self.crop_data.get(crop_type, self.crop_data['wheat'])
        priority_machines = crop_info['mechanization_priority']
        
        recommendations = []
        
        if mechanization_level == 'low' and area > 1.0:
            # Primary mechanization
            recommendations.append({
                'category': 'Primary Mechanization',
                'priority': 'High',
                'recommendation': f'Invest in {priority_machines[0]} for improved efficiency',
                'expected_benefit': f'Reduce labor costs by 60% (₹{area * 8000:.0f}/season)',
                'implementation_cost': f'₹{150000 if priority_machines[0] == "combine_harvester" else 80000}',
                'payback_period': '3-4 seasons',
                'action_items': [
                    f'Purchase or lease {priority_machines[0]}',
                    'Train operators for safe operation',
                    'Form farmer groups for shared ownership',
                    'Access government subsidies (40-50% available)'
                ]
            })
        
        # Precision agriculture
        if area > 2.0:
            recommendations.append({
                'category': 'Precision Agriculture',
                'priority': 'Medium',
                'recommendation': 'Adopt precision agriculture technologies for data-driven farming',
                'expected_benefit': '+12-15% input efficiency, +8-10% yield improvement',
                'implementation_cost': f'₹{area * 5000:.0f}',
                'payback_period': '2-3 seasons',
                'action_items': [
                    'Install soil moisture sensors',
                    'Use GPS-guided machinery',
                    'Implement variable rate application',
                    'Adopt farm management software'
                ]
            })
        
        # Digital tools
        recommendations.append({
            'category': 'Digital Agriculture',
            'priority': 'Low',
            'recommendation': 'Leverage digital tools for farm monitoring and decision-making',
            'expected_benefit': 'Better timing of operations, early problem detection',
            'implementation_cost': '₹5,000-10,000',
            'payback_period': '1 season',
            'action_items': [
                'Download and use farming apps (e.g., KisanSuvidha, AgriApp)',
                'Join WhatsApp groups for weather and market updates',
                'Use satellite imagery for crop monitoring',
                'Maintain digital farm records'
            ]
        })
        
        return recommendations
    
    def _get_cost_optimization_recommendations(self, farm_data: Dict) -> List[Dict]:
        """Generate cost optimization and market alignment recommendations"""
        area = float(farm_data.get('area_hectares', 2.5))
        crop_type = farm_data.get('crop_type', 'wheat').lower()
        current_yield = farm_data.get('yield_kg_per_ha', 3000)
        
        market_info = self.market_prices.get(crop_type, self.market_prices['wheat'])
        
        recommendations = []
        
        # Input cost reduction
        recommendations.append({
            'category': 'Input Cost Optimization',
            'priority': 'High',
            'recommendation': 'Reduce input costs through bulk purchasing and alternative sources',
            'expected_benefit': f'Save 15-20% on input costs (₹{area * 3000:.0f})',
            'implementation_cost': 'Minimal',
            'payback_period': 'Immediate',
            'action_items': [
                'Form/join Farmer Producer Organizations (FPOs)',
                'Buy inputs directly from manufacturers',
                'Use government subsidized inputs',
                'Compare prices from multiple vendors'
            ]
        })
        
        # Market linkage
        recommendations.append({
            'category': 'Market Linkage',
            'priority': 'High',
            'recommendation': f'Access better markets to get ₹{market_info["market_avg"] - 2000}/quintal instead of local rates',
            'expected_benefit': f'+₹{(market_info["market_avg"] - 2000) * current_yield * area / 100:.0f} additional income',
            'implementation_cost': f'₹{area * 1000:.0f}',
            'payback_period': '1 season',
            'action_items': [
                'Register on eNAM platform',
                'Join commodity exchanges',
                'Connect with food processing companies',
                'Use mobile apps for price discovery'
            ]
        })
        
        # Value addition
        if area > 3.0:
            recommendations.append({
                'category': 'Value Addition',
                'priority': 'Medium',
                'recommendation': 'Explore value addition opportunities for higher margins',
                'expected_benefit': f'+25-40% price premium (₹{current_yield * area * 5:.0f})',
                'implementation_cost': f'₹{area * 10000:.0f}',
                'payback_period': '2-3 seasons',
                'action_items': [
                    'Set up small processing unit',
                    'Obtain food safety certifications',
                    'Develop brand and packaging',
                    'Market directly to consumers'
                ]
            })
        
        return recommendations
    
    def _get_sustainability_recommendations(self, farm_data: Dict) -> List[Dict]:
        """Generate sustainability improvement recommendations"""
        area = float(farm_data.get('area_hectares', 2.5))
        sustainability_score = farm_data.get('sustainability_score', 75)
        
        recommendations = []
        
        # Carbon farming
        if sustainability_score < 85:
            recommendations.append({
                'category': 'Carbon Sequestration',
                'priority': 'Medium',
                'recommendation': 'Implement carbon farming practices for environmental and economic benefits',
                'expected_benefit': f'Potential ₹{area * 2000:.0f}/year from carbon credits',
                'implementation_cost': f'₹{area * 3000:.0f}',
                'payback_period': '2-3 seasons',
                'action_items': [
                    'Practice conservation tillage',
                    'Plant trees on farm boundaries',
                    'Use cover crops and crop residue incorporation',
                    'Register for carbon credit programs'
                ]
            })
        
        # Biodiversity enhancement
        recommendations.append({
            'category': 'Biodiversity Conservation',
            'priority': 'Low',
            'recommendation': 'Enhance on-farm biodiversity for ecosystem resilience',
            'expected_benefit': 'Natural pest control, pollination services, soil health',
            'implementation_cost': f'₹{area * 1500:.0f}',
            'payback_period': '3-5 seasons',
            'action_items': [
                'Plant native trees and shrubs',
                'Create pollinator habitats',
                'Maintain crop diversity',
                'Avoid monoculture practices'
            ]
        })
        
        # Resource efficiency
        recommendations.append({
            'category': 'Resource Efficiency',
            'priority': 'High',
            'recommendation': 'Improve resource use efficiency for sustainable intensification',
            'expected_benefit': 'Reduced environmental impact, lower costs',
            'implementation_cost': f'₹{area * 2500:.0f}',
            'payback_period': '2 seasons',
            'action_items': [
                'Adopt 4R nutrient stewardship (Right source, rate, time, place)',
                'Use renewable energy (solar pumps)',
                'Practice integrated farming systems',
                'Monitor and reduce greenhouse gas emissions'
            ]
        })
        
        return recommendations
    
    def _get_market_strategy_recommendations(self, farm_data: Dict) -> List[Dict]:
        """Generate market strategy and timing recommendations"""
        crop_type = farm_data.get('crop_type', 'wheat').lower()
        area = float(farm_data.get('area_hectares', 2.5))
        current_yield = farm_data.get('yield_kg_per_ha', 3000)
        
        market_info = self.market_prices.get(crop_type, self.market_prices['wheat'])
        
        recommendations = []
        
        # Price discovery
        recommendations.append({
            'category': 'Price Intelligence',
            'priority': 'High',
            'recommendation': 'Use real-time price information for optimal selling decisions',
            'expected_benefit': f'+5-10% better prices (₹{current_yield * area * 2:.0f})',
            'implementation_cost': '₹2,000/year',
            'payback_period': 'Immediate',
            'action_items': [
                'Subscribe to price alerts (SMS/WhatsApp)',
                'Use AgMarkNet mobile app',
                'Monitor international commodity prices',
                'Track seasonal price patterns'
            ]
        })
        
        # Storage and timing
        recommendations.append({
            'category': 'Post-Harvest Management',
            'priority': 'Medium',
            'recommendation': 'Improve storage to capitalize on seasonal price variations',
            'expected_benefit': f'+15-25% price advantage (₹{current_yield * area * 4:.0f})',
            'implementation_cost': f'₹{area * 8000:.0f}',
            'payback_period': '2-3 seasons',
            'action_items': [
                'Invest in proper storage facilities',
                'Use scientific storage methods',
                'Obtain warehouse receipts',
                'Plan harvest timing strategically'
            ]
        })
        
        return recommendations
    
    def _get_priority_actions(self, farm_data: Dict) -> List[Dict]:
        """Get prioritized action plan based on impact and feasibility"""
        all_recommendations = []
        
        # Collect all recommendations
        for category in ['crop_management', 'resource_optimization', 'soil_health', 
                        'mechanization', 'cost_optimization']:
            method = getattr(self, f'_get_{category}_recommendations')
            all_recommendations.extend(method(farm_data))
        
        # Sort by priority and expected benefit
        priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
        all_recommendations.sort(key=lambda x: priority_map.get(x['priority'], 0), reverse=True)
        
        # Return top 5 priority actions
        return all_recommendations[:5]
    
    def _calculate_roi_projections(self, farm_data: Dict) -> Dict:
        """Calculate ROI projections for recommended improvements"""
        area = float(farm_data.get('area_hectares', 2.5))
        current_yield = farm_data.get('yield_kg_per_ha', 3000)
        crop_type = farm_data.get('crop_type', 'wheat').lower()
        
        market_info = self.market_prices.get(crop_type, self.market_prices['wheat'])
        current_price = market_info['market_avg']
        
        # Current economics
        current_revenue = current_yield * area * current_price / 100
        current_costs = area * 25000  # Estimated current costs
        current_profit = current_revenue - current_costs
        
        # Projected improvements
        improved_yield = current_yield * 1.2  # 20% yield improvement
        improved_price = current_price * 1.1   # 10% better price
        improved_costs = current_costs * 1.05  # 5% cost increase
        
        improved_revenue = improved_yield * area * improved_price / 100
        improved_profit = improved_revenue - improved_costs
        
        additional_profit = improved_profit - current_profit
        roi_percentage = (additional_profit / (improved_costs - current_costs)) * 100
        
        return {
            'current_profit': current_profit,
            'projected_profit': improved_profit,
            'additional_profit': additional_profit,
            'roi_percentage': roi_percentage,
            'payback_period': '1.5-2.0 seasons',
            'investment_required': improved_costs - current_costs,
            'profit_improvement': ((improved_profit - current_profit) / current_profit) * 100
        }

def create_recommendations_summary(recommendations: Dict) -> str:
    """Create a formatted summary of recommendations"""
    
    summary = []
    summary.append("# 🎯 COMPREHENSIVE FARM OPTIMIZATION RECOMMENDATIONS\n")
    
    # Priority actions
    summary.append("## 🔥 TOP PRIORITY ACTIONS\n")
    for i, action in enumerate(recommendations['priority_actions'], 1):
        summary.append(f"**{i}. {action['category']}** - {action['priority']} Priority")
        summary.append(f"   {action['recommendation']}")
        summary.append(f"   💰 Expected Benefit: {action['expected_benefit']}")
        summary.append(f"   💸 Investment: {action['implementation_cost']}")
        summary.append("")
    
    # ROI Summary
    roi = recommendations['roi_projections']
    summary.append("## 📈 RETURN ON INVESTMENT PROJECTION\n")
    summary.append(f"- **Current Annual Profit**: ₹{roi['current_profit']:,.0f}")
    summary.append(f"- **Projected Annual Profit**: ₹{roi['projected_profit']:,.0f}")
    summary.append(f"- **Additional Profit**: ₹{roi['additional_profit']:,.0f}")
    summary.append(f"- **Profit Improvement**: {roi['profit_improvement']:.1f}%")
    summary.append(f"- **ROI**: {roi['roi_percentage']:.1f}%")
    summary.append(f"- **Payback Period**: {roi['payback_period']}\n")
    
    return "\n".join(summary)