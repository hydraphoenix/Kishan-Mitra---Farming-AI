"""
Data Generation Utilities for AgroMRV System
Generate realistic agricultural MRV data for 15 representative farms
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import logging
from ..models.mrv_node import SmallholderMRVNode, create_demo_farms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MRVDataGenerator:
    """Comprehensive data generator for agricultural MRV datasets"""
    
    def __init__(self):
        self.demo_farms = create_demo_farms()
        logger.info(f"Initialized data generator with {len(self.demo_farms)} farms")
    
    def generate_comprehensive_dataset(self, days: int = 60) -> pd.DataFrame:
        """
        Generate comprehensive MRV dataset for all demo farms
        
        Args:
            days: Number of days of historical data per farm
            
        Returns:
            Combined DataFrame with all farm data
        """
        all_data = []
        
        logger.info(f"Generating {days} days of data for {len(self.demo_farms)} farms")
        
        for i, farm in enumerate(self.demo_farms):
            try:
                farm_data = farm.generate_historical_data(days)
                all_data.append(farm_data)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Generated data for {i + 1}/{len(self.demo_farms)} farms")
                    
            except Exception as e:
                logger.error(f"Error generating data for farm {farm.farm_id}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data generated - all farms failed")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Generated total of {len(combined_data)} records")
        
        return combined_data
    
    def generate_farm_comparison_data(self) -> Dict:
        """Generate data for comparing farms across different metrics"""
        
        comparison_data = {
            'farm_summaries': [],
            'state_aggregates': {},
            'crop_aggregates': {},
            'performance_rankings': {}
        }
        
        # Generate 30 days of data for comparison
        farm_data = []
        
        for farm in self.demo_farms:
            try:
                data = farm.generate_historical_data(30)
                
                # Calculate farm summary metrics
                summary = {
                    'farm_id': farm.farm_id,
                    'state': farm.state,
                    'crop_type': farm.crop_type,
                    'area_hectares': farm.area_hectares,
                    'avg_carbon_sequestration': data['co2_sequestered_kg'].mean(),
                    'avg_emissions': data['co2_emissions_kg'].mean(),
                    'avg_net_carbon': data['net_carbon_balance_kg'].mean(),
                    'avg_yield': data['yield_kg_per_ha'].mean(),
                    'avg_sustainability': data['sustainability_score'].mean(),
                    'total_carbon_credits': data['carbon_credits_potential'].sum(),
                    'water_efficiency': data['water_efficiency'].mean()
                }
                
                comparison_data['farm_summaries'].append(summary)
                farm_data.append(data)
                
            except Exception as e:
                logger.error(f"Error in comparison data for {farm.farm_id}: {e}")
                continue
        
        # State-wise aggregation
        df = pd.DataFrame(comparison_data['farm_summaries'])
        
        for state in df['state'].unique():
            state_data = df[df['state'] == state]
            comparison_data['state_aggregates'][state] = {
                'farm_count': len(state_data),
                'total_area': state_data['area_hectares'].sum(),
                'avg_carbon_sequestration': state_data['avg_carbon_sequestration'].mean(),
                'avg_sustainability': state_data['avg_sustainability'].mean(),
                'total_carbon_credits': state_data['total_carbon_credits'].sum()
            }
        
        # Crop-wise aggregation
        for crop in df['crop_type'].unique():
            crop_data = df[df['crop_type'] == crop]
            comparison_data['crop_aggregates'][crop] = {
                'farm_count': len(crop_data),
                'avg_yield': crop_data['avg_yield'].mean(),
                'avg_net_carbon': crop_data['avg_net_carbon'].mean(),
                'avg_water_efficiency': crop_data['water_efficiency'].mean()
            }
        
        # Performance rankings
        comparison_data['performance_rankings'] = {
            'top_carbon_sequestration': df.nlargest(5, 'avg_carbon_sequestration')[['farm_id', 'avg_carbon_sequestration']].to_dict('records'),
            'top_sustainability': df.nlargest(5, 'avg_sustainability')[['farm_id', 'avg_sustainability']].to_dict('records'),
            'top_yield': df.nlargest(5, 'avg_yield')[['farm_id', 'avg_yield']].to_dict('records'),
            'most_carbon_credits': df.nlargest(5, 'total_carbon_credits')[['farm_id', 'total_carbon_credits']].to_dict('records')
        }
        
        logger.info("Generated comprehensive farm comparison data")
        return comparison_data
    
    def generate_seasonal_analysis(self) -> Dict:
        """Generate seasonal analysis data for agricultural patterns"""
        
        seasonal_data = {
            'monthly_trends': {},
            'seasonal_patterns': {},
            'climate_correlations': {}
        }
        
        # Generate full year of data for seasonal analysis
        all_farm_data = []
        
        for farm in self.demo_farms[:8]:  # Use subset for performance
            try:
                year_data = farm.generate_historical_data(365)
                year_data['month'] = pd.to_datetime(year_data['date']).dt.month
                year_data['season'] = year_data['month'].map(self._get_season)
                all_farm_data.append(year_data)
            except Exception as e:
                logger.error(f"Error generating seasonal data for {farm.farm_id}: {e}")
                continue
        
        if all_farm_data:
            combined_seasonal = pd.concat(all_farm_data, ignore_index=True)
            
            # Monthly trends
            monthly_agg = combined_seasonal.groupby('month').agg({
                'co2_sequestered_kg': 'mean',
                'net_carbon_balance_kg': 'mean',
                'yield_kg_per_ha': 'mean',
                'sustainability_score': 'mean',
                'temperature_celsius': 'mean',
                'rainfall_mm': 'mean'
            }).round(2)
            
            seasonal_data['monthly_trends'] = monthly_agg.to_dict('index')
            
            # Seasonal patterns
            seasonal_agg = combined_seasonal.groupby('season').agg({
                'co2_sequestered_kg': ['mean', 'std'],
                'yield_kg_per_ha': ['mean', 'std'],
                'water_usage_liters': ['mean', 'std'],
                'sustainability_score': ['mean', 'std']
            }).round(2)
            
            seasonal_data['seasonal_patterns'] = seasonal_agg.to_dict()
            
            # Climate correlations
            climate_corr = combined_seasonal[['temperature_celsius', 'humidity_percent', 'rainfall_mm', 
                                           'co2_sequestered_kg', 'yield_kg_per_ha', 'sustainability_score']].corr()
            
            seasonal_data['climate_correlations'] = climate_corr.to_dict()
        
        logger.info("Generated seasonal analysis data")
        return seasonal_data
    
    def _get_season(self, month: int) -> str:
        """Map month to Indian agricultural season"""
        if month in [6, 7, 8, 9, 10]:  # June-October
            return 'Kharif'
        elif month in [11, 12, 1, 2, 3]:  # November-March
            return 'Rabi'
        else:  # April-May
            return 'Summer'
    
    def generate_carbon_credit_analysis(self) -> Dict:
        """Generate carbon credit potential analysis"""
        
        credit_analysis = {
            'farm_credit_potential': [],
            'state_totals': {},
            'crop_efficiency': {},
            'market_projections': {}
        }
        
        for farm in self.demo_farms:
            try:
                data = farm.generate_historical_data(90)  # 3 months
                
                total_credits = data['carbon_credits_potential'].sum()
                avg_daily_credits = data['carbon_credits_potential'].mean()
                
                farm_credits = {
                    'farm_id': farm.farm_id,
                    'state': farm.state,
                    'crop_type': farm.crop_type,
                    'area_hectares': farm.area_hectares,
                    'total_credits_3months': round(total_credits, 4),
                    'daily_avg_credits': round(avg_daily_credits, 4),
                    'annual_projection': round(total_credits * 4, 4),  # Project to full year
                    'credits_per_hectare': round(total_credits / farm.area_hectares, 4),
                    'carbon_efficiency': round(data['net_carbon_balance_kg'].mean(), 2)
                }
                
                credit_analysis['farm_credit_potential'].append(farm_credits)
                
            except Exception as e:
                logger.error(f"Error in carbon credit analysis for {farm.farm_id}: {e}")
                continue
        
        # State-wise totals
        df = pd.DataFrame(credit_analysis['farm_credit_potential'])
        
        for state in df['state'].unique():
            state_data = df[df['state'] == state]
            credit_analysis['state_totals'][state] = {
                'total_farms': len(state_data),
                'total_annual_credits': round(state_data['annual_projection'].sum(), 4),
                'avg_credits_per_farm': round(state_data['annual_projection'].mean(), 4),
                'total_area_hectares': round(state_data['area_hectares'].sum(), 2)
            }
        
        # Crop efficiency analysis
        for crop in df['crop_type'].unique():
            crop_data = df[df['crop_type'] == crop]
            credit_analysis['crop_efficiency'][crop] = {
                'avg_credits_per_hectare': round(crop_data['credits_per_hectare'].mean(), 4),
                'avg_carbon_efficiency': round(crop_data['carbon_efficiency'].mean(), 2),
                'farm_count': len(crop_data)
            }
        
        # Market projections
        total_annual_credits = df['annual_projection'].sum()
        credit_price_inr = 2000  # ₹2000 per credit (estimated)
        
        credit_analysis['market_projections'] = {
            'total_annual_credits_all_farms': round(total_annual_credits, 4),
            'estimated_market_value_inr': round(total_annual_credits * credit_price_inr, 2),
            'avg_revenue_per_farm_inr': round((total_annual_credits * credit_price_inr) / len(df), 2),
            'credit_price_assumed_inr': credit_price_inr
        }
        
        logger.info("Generated carbon credit analysis")
        return credit_analysis
    
    def generate_sustainability_report(self) -> Dict:
        """Generate comprehensive sustainability assessment"""
        
        sustainability_report = {
            'overall_metrics': {},
            'farm_rankings': [],
            'improvement_areas': {},
            'best_practices': {}
        }
        
        all_sustainability_data = []
        
        for farm in self.demo_farms:
            try:
                data = farm.generate_historical_data(30)
                
                farm_sustainability = {
                    'farm_id': farm.farm_id,
                    'state': farm.state,
                    'crop_type': farm.crop_type,
                    'overall_sustainability': data['sustainability_score'].mean(),
                    'soil_health': data['soil_health_index'].mean(),
                    'water_efficiency': data['water_efficiency'].mean(),
                    'biodiversity': data['biodiversity_index'].mean(),
                    'carbon_balance': data['net_carbon_balance_kg'].mean(),
                    'resource_efficiency': (data['yield_kg_per_ha'] / data['water_usage_liters'] * 1000).mean()
                }
                
                all_sustainability_data.append(farm_sustainability)
                
            except Exception as e:
                logger.error(f"Error in sustainability analysis for {farm.farm_id}: {e}")
                continue
        
        df = pd.DataFrame(all_sustainability_data)
        
        # Overall metrics
        sustainability_report['overall_metrics'] = {
            'avg_sustainability_score': round(df['overall_sustainability'].mean(), 2),
            'avg_soil_health': round(df['soil_health'].mean(), 2),
            'avg_water_efficiency': round(df['water_efficiency'].mean(), 2),
            'avg_biodiversity': round(df['biodiversity'].mean(), 2),
            'farms_above_80_sustainability': len(df[df['overall_sustainability'] >= 80]),
            'total_farms_assessed': len(df)
        }
        
        # Farm rankings
        sustainability_report['farm_rankings'] = df.nlargest(10, 'overall_sustainability')[
            ['farm_id', 'state', 'crop_type', 'overall_sustainability']
        ].to_dict('records')
        
        # Improvement areas (farms below average)
        below_avg = df[df['overall_sustainability'] < df['overall_sustainability'].mean()]
        if len(below_avg) > 0:
            sustainability_report['improvement_areas'] = {
                'farms_needing_improvement': len(below_avg),
                'avg_soil_health_deficit': round(80 - below_avg['soil_health'].mean(), 2),
                'avg_water_efficiency_deficit': round(80 - below_avg['water_efficiency'].mean(), 2),
                'priority_states': below_avg['state'].value_counts().to_dict()
            }
        
        # Best practices (top performers)
        top_performers = df.nlargest(5, 'overall_sustainability')
        if len(top_performers) > 0:
            sustainability_report['best_practices'] = {
                'leading_states': top_performers['state'].value_counts().to_dict(),
                'leading_crops': top_performers['crop_type'].value_counts().to_dict(),
                'avg_metrics_top_performers': {
                    'sustainability': round(top_performers['overall_sustainability'].mean(), 2),
                    'soil_health': round(top_performers['soil_health'].mean(), 2),
                    'water_efficiency': round(top_performers['water_efficiency'].mean(), 2)
                }
            }
        
        logger.info("Generated comprehensive sustainability report")
        return sustainability_report
    
    def save_generated_data(self, output_dir: str = "generated_data") -> Dict[str, str]:
        """Save all generated data to files"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        try:
            # Main dataset
            main_data = self.generate_comprehensive_dataset(60)
            main_file = os.path.join(output_dir, "mrv_dataset_60days.csv")
            main_data.to_csv(main_file, index=False)
            saved_files['main_dataset'] = main_file
            
            # Farm comparison
            comparison_data = self.generate_farm_comparison_data()
            comp_file = os.path.join(output_dir, "farm_comparison.json")
            with open(comp_file, 'w') as f:
                json.dump(comparison_data, f, indent=2)
            saved_files['farm_comparison'] = comp_file
            
            # Seasonal analysis
            seasonal_data = self.generate_seasonal_analysis()
            seasonal_file = os.path.join(output_dir, "seasonal_analysis.json")
            with open(seasonal_file, 'w') as f:
                json.dump(seasonal_data, f, indent=2, default=str)
            saved_files['seasonal_analysis'] = seasonal_file
            
            # Carbon credits
            credit_data = self.generate_carbon_credit_analysis()
            credit_file = os.path.join(output_dir, "carbon_credit_analysis.json")
            with open(credit_file, 'w') as f:
                json.dump(credit_data, f, indent=2)
            saved_files['carbon_credits'] = credit_file
            
            # Sustainability report
            sustainability_data = self.generate_sustainability_report()
            sustain_file = os.path.join(output_dir, "sustainability_report.json")
            with open(sustain_file, 'w') as f:
                json.dump(sustainability_data, f, indent=2)
            saved_files['sustainability_report'] = sustain_file
            
            logger.info(f"All data saved to {output_dir}/")
            
        except Exception as e:
            logger.error(f"Error saving generated data: {e}")
            raise
        
        return saved_files

def create_sample_datasets():
    """Create sample datasets for testing and demo"""
    
    generator = MRVDataGenerator()
    
    # Quick dataset for testing (10 days)
    quick_data = generator.generate_comprehensive_dataset(10)
    
    # Sample of different data types
    samples = {
        'quick_dataset': quick_data,
        'farm_comparison': generator.generate_farm_comparison_data(),
        'carbon_credits': generator.generate_carbon_credit_analysis()
    }
    
    logger.info("Created sample datasets for testing")
    return samples

if __name__ == "__main__":
    # Demo data generation
    generator = MRVDataGenerator()
    
    # Generate and save all data
    saved_files = generator.save_generated_data()
    
    print("=== Data Generation Complete ===")
    for data_type, filepath in saved_files.items():
        print(f"{data_type}: {filepath}")
    
    # Show sample data statistics
    main_data = generator.generate_comprehensive_dataset(30)
    print(f"\nMain dataset: {len(main_data)} records from {main_data['farm_id'].nunique()} farms")
    print(f"Date range: {main_data['date'].min()} to {main_data['date'].max()}")
    print(f"States covered: {main_data['state'].nunique()}")
    print(f"Crop types: {main_data['crop_type'].nunique()}")