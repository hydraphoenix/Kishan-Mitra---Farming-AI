"""
Data Processing Utilities for AgroMRV System
Advanced data processing, validation, and transformation utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MRVDataProcessor:
    """Comprehensive data processor for MRV datasets"""
    
    def __init__(self):
        self.scalers = {}
        self.validation_rules = self._define_validation_rules()
        self.processed_stats = {}
    
    def _define_validation_rules(self) -> Dict:
        """Define validation rules for MRV data quality"""
        return {
            'temperature_celsius': {'min': -10, 'max': 55, 'type': 'float'},
            'humidity_percent': {'min': 10, 'max': 100, 'type': 'float'},
            'rainfall_mm': {'min': 0, 'max': 500, 'type': 'float'},
            'soil_ph': {'min': 3.0, 'max': 10.0, 'type': 'float'},
            'soil_organic_carbon_percent': {'min': 0.1, 'max': 5.0, 'type': 'float'},
            'soil_nitrogen_kg_per_ha': {'min': 50, 'max': 1000, 'type': 'float'},
            'soil_phosphorus_kg_per_ha': {'min': 5, 'max': 200, 'type': 'float'},
            'water_usage_liters': {'min': 10, 'max': 10000, 'type': 'float'},
            'fertilizer_n_kg': {'min': 0, 'max': 500, 'type': 'float'},
            'fertilizer_p_kg': {'min': 0, 'max': 200, 'type': 'float'},
            'pesticide_kg': {'min': 0, 'max': 50, 'type': 'float'},
            'yield_kg_per_ha': {'min': 100, 'max': 100000, 'type': 'float'},
            'sustainability_score': {'min': 0, 'max': 100, 'type': 'float'},
            'area_hectares': {'min': 0.1, 'max': 50, 'type': 'float'}
        }
    
    def validate_data(self, data: pd.DataFrame) -> Dict:
        """
        Comprehensive data validation with quality assessment
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            Dictionary with validation results and quality metrics
        """
        validation_results = {
            'total_records': len(data),
            'valid_records': 0,
            'validation_errors': [],
            'quality_score': 0,
            'field_quality': {},
            'anomalies_detected': [],
            'recommendations': []
        }
        
        # Check required columns
        required_columns = [
            'date', 'farm_id', 'state', 'crop_type', 'area_hectares',
            'temperature_celsius', 'humidity_percent', 'rainfall_mm',
            'co2_sequestered_kg', 'net_carbon_balance_kg', 'yield_kg_per_ha'
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_results['validation_errors'].append(
                f"Missing required columns: {missing_columns}"
            )
            return validation_results
        
        # Validate each field
        valid_rows = np.ones(len(data), dtype=bool)
        
        for field, rules in self.validation_rules.items():
            if field in data.columns:
                field_valid = np.ones(len(data), dtype=bool)
                
                # Type validation
                try:
                    if rules['type'] == 'float':
                        pd.to_numeric(data[field], errors='coerce')
                except:
                    validation_results['validation_errors'].append(
                        f"Invalid data type in field: {field}"
                    )
                    continue
                
                # Range validation
                field_data = pd.to_numeric(data[field], errors='coerce')
                
                # Check for NaN values
                nan_count = field_data.isna().sum()
                if nan_count > 0:
                    field_valid &= ~field_data.isna()
                    validation_results['validation_errors'].append(
                        f"Found {nan_count} NaN values in {field}"
                    )
                
                # Range checks
                below_min = field_data < rules['min']
                above_max = field_data > rules['max']
                
                if below_min.sum() > 0:
                    field_valid &= ~below_min
                    validation_results['validation_errors'].append(
                        f"Found {below_min.sum()} values below minimum ({rules['min']}) in {field}"
                    )
                
                if above_max.sum() > 0:
                    field_valid &= ~above_max
                    validation_results['validation_errors'].append(
                        f"Found {above_max.sum()} values above maximum ({rules['max']}) in {field}"
                    )
                
                # Field quality score
                field_quality = field_valid.sum() / len(data) * 100
                validation_results['field_quality'][field] = round(field_quality, 2)
                
                # Update overall validity
                valid_rows &= field_valid
        
        # Detect statistical anomalies
        anomalies = self._detect_anomalies(data)
        validation_results['anomalies_detected'] = anomalies
        
        # Calculate overall quality scores
        validation_results['valid_records'] = valid_rows.sum()
        validation_results['quality_score'] = round(
            (validation_results['valid_records'] / validation_results['total_records']) * 100, 2
        )
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_recommendations(validation_results)
        
        logger.info(f"Data validation complete - Quality Score: {validation_results['quality_score']}%")
        return validation_results
    
    def _detect_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """Detect statistical anomalies in the data"""
        anomalies = []
        
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_columns:
            if col in self.validation_rules:
                try:
                    # Z-score based anomaly detection
                    z_scores = np.abs(stats.zscore(data[col].dropna()))
                    anomaly_threshold = 3
                    
                    anomaly_count = (z_scores > anomaly_threshold).sum()
                    
                    if anomaly_count > 0:
                        anomalies.append({
                            'field': col,
                            'anomaly_type': 'statistical_outlier',
                            'count': int(anomaly_count),
                            'threshold': anomaly_threshold,
                            'severity': 'high' if anomaly_count > len(data) * 0.05 else 'medium'
                        })
                
                except Exception as e:
                    logger.warning(f"Could not detect anomalies in {col}: {e}")
                    continue
        
        return anomalies
    
    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []
        
        # Quality score based recommendations
        if validation_results['quality_score'] < 80:
            recommendations.append("Data quality is below recommended threshold (80%). Consider data cleansing.")
        
        # Field-specific recommendations
        for field, quality in validation_results['field_quality'].items():
            if quality < 90:
                recommendations.append(f"Field '{field}' has quality issues ({quality}%). Review data collection for this parameter.")
        
        # Anomaly-based recommendations
        high_severity_anomalies = [a for a in validation_results['anomalies_detected'] if a['severity'] == 'high']
        if high_severity_anomalies:
            recommendations.append("High severity anomalies detected. Review data collection protocols.")
        
        return recommendations
    
    def clean_data(self, data: pd.DataFrame, cleaning_strategy: str = 'moderate') -> pd.DataFrame:
        """
        Clean and preprocess MRV data
        
        Args:
            data: Input DataFrame
            cleaning_strategy: 'conservative', 'moderate', or 'aggressive'
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        cleaning_log = []
        
        # Remove duplicate records
        duplicates = cleaned_data.duplicated()
        if duplicates.sum() > 0:
            cleaned_data = cleaned_data.drop_duplicates()
            cleaning_log.append(f"Removed {duplicates.sum()} duplicate records")
        
        # Handle missing values
        for column in cleaned_data.columns:
            if column in self.validation_rules:
                missing_count = cleaned_data[column].isna().sum()
                
                if missing_count > 0:
                    if cleaning_strategy == 'conservative':
                        # Fill with median for numerical columns
                        if cleaned_data[column].dtype in ['float64', 'int64']:
                            cleaned_data[column].fillna(cleaned_data[column].median(), inplace=True)
                    
                    elif cleaning_strategy == 'moderate':
                        # Use forward fill then median
                        cleaned_data[column] = cleaned_data[column].fillna(method='ffill')
                        cleaned_data[column].fillna(cleaned_data[column].median(), inplace=True)
                    
                    elif cleaning_strategy == 'aggressive':
                        # Remove rows with missing critical values
                        critical_columns = ['date', 'farm_id', 'co2_sequestered_kg', 'yield_kg_per_ha']
                        if column in critical_columns:
                            cleaned_data = cleaned_data.dropna(subset=[column])
                        else:
                            cleaned_data[column].fillna(cleaned_data[column].median(), inplace=True)
                    
                    cleaning_log.append(f"Handled {missing_count} missing values in {column}")
        
        # Cap outliers
        for column in cleaned_data.select_dtypes(include=[np.number]).columns:
            if column in self.validation_rules:
                rules = self.validation_rules[column]
                
                # Cap values to valid range
                original_out_of_range = ((cleaned_data[column] < rules['min']) | 
                                       (cleaned_data[column] > rules['max'])).sum()
                
                if original_out_of_range > 0:
                    cleaned_data[column] = np.clip(cleaned_data[column], rules['min'], rules['max'])
                    cleaning_log.append(f"Capped {original_out_of_range} out-of-range values in {column}")
        
        # Statistical outlier handling
        if cleaning_strategy in ['moderate', 'aggressive']:
            for column in cleaned_data.select_dtypes(include=[np.number]).columns:
                if column in self.validation_rules:
                    # Use IQR method for outlier detection
                    Q1 = cleaned_data[column].quantile(0.25)
                    Q3 = cleaned_data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((cleaned_data[column] < lower_bound) | 
                              (cleaned_data[column] > upper_bound)).sum()
                    
                    if outliers > 0:
                        if cleaning_strategy == 'moderate':
                            # Cap outliers to bounds
                            cleaned_data[column] = np.clip(cleaned_data[column], lower_bound, upper_bound)
                        elif cleaning_strategy == 'aggressive':
                            # Remove outlier rows
                            cleaned_data = cleaned_data[
                                (cleaned_data[column] >= lower_bound) & 
                                (cleaned_data[column] <= upper_bound)
                            ]
                        
                        cleaning_log.append(f"Handled {outliers} statistical outliers in {column}")
        
        self.processed_stats['cleaning_log'] = cleaning_log
        self.processed_stats['records_before_cleaning'] = len(data)
        self.processed_stats['records_after_cleaning'] = len(cleaned_data)
        
        logger.info(f"Data cleaning complete: {len(data)} -> {len(cleaned_data)} records")
        return cleaned_data
    
    def normalize_data(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize numerical columns in the dataset
        
        Args:
            data: Input DataFrame
            method: 'standard', 'minmax', or 'robust'
            
        Returns:
            Normalized DataFrame
        """
        normalized_data = data.copy()
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        
        # Exclude certain columns from normalization
        exclude_columns = ['date', 'area_hectares', 'yield_kg_per_ha']
        columns_to_normalize = [col for col in numerical_columns if col not in exclude_columns]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Supported methods: 'standard', 'minmax'")
        
        # Fit and transform
        normalized_data[columns_to_normalize] = scaler.fit_transform(
            normalized_data[columns_to_normalize]
        )
        
        # Store scaler for future use
        self.scalers[method] = scaler
        
        logger.info(f"Data normalization complete using {method} method")
        return normalized_data
    
    def aggregate_data(self, data: pd.DataFrame, aggregation_level: str = 'farm') -> pd.DataFrame:
        """
        Aggregate MRV data at different levels
        
        Args:
            data: Input DataFrame
            aggregation_level: 'farm', 'state', 'crop', or 'monthly'
            
        Returns:
            Aggregated DataFrame
        """
        agg_functions = {
            'co2_sequestered_kg': ['mean', 'sum', 'std'],
            'co2_emissions_kg': ['mean', 'sum'],
            'net_carbon_balance_kg': ['mean', 'sum'],
            'yield_kg_per_ha': ['mean', 'std'],
            'sustainability_score': ['mean', 'std'],
            'water_usage_liters': ['mean', 'sum'],
            'carbon_credits_potential': ['sum'],
            'temperature_celsius': ['mean'],
            'rainfall_mm': ['sum', 'mean']
        }
        
        if aggregation_level == 'farm':
            grouped = data.groupby('farm_id')
        elif aggregation_level == 'state':
            grouped = data.groupby('state')
        elif aggregation_level == 'crop':
            grouped = data.groupby('crop_type')
        elif aggregation_level == 'monthly':
            data['month_year'] = pd.to_datetime(data['date']).dt.to_period('M')
            grouped = data.groupby('month_year')
        else:
            raise ValueError("Supported levels: 'farm', 'state', 'crop', 'monthly'")
        
        # Perform aggregation
        aggregated = grouped.agg(agg_functions).round(3)
        
        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
        
        logger.info(f"Data aggregated at {aggregation_level} level")
        return aggregated
    
    def calculate_derived_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional derived metrics from MRV data"""
        enhanced_data = data.copy()
        
        # Carbon efficiency metrics
        enhanced_data['carbon_efficiency'] = (
            enhanced_data['co2_sequestered_kg'] / 
            (enhanced_data['co2_emissions_kg'] + enhanced_data['n2o_emissions_kg'] + enhanced_data['ch4_emissions_kg'])
        ).fillna(0)
        
        # Water productivity (yield per water unit)
        enhanced_data['water_productivity'] = (
            enhanced_data['yield_kg_per_ha'] / enhanced_data['water_usage_liters'] * 1000
        ).fillna(0)
        
        # Fertilizer efficiency
        enhanced_data['fertilizer_efficiency'] = (
            enhanced_data['yield_kg_per_ha'] / (enhanced_data['fertilizer_n_kg'] + 1)
        ).fillna(0)
        
        # Environmental impact score (lower is better)
        enhanced_data['environmental_impact'] = (
            (enhanced_data['co2_emissions_kg'] + enhanced_data['n2o_emissions_kg'] + enhanced_data['ch4_emissions_kg']) /
            enhanced_data['area_hectares']
        ).fillna(0)
        
        # Carbon credit efficiency
        enhanced_data['carbon_credit_efficiency'] = (
            enhanced_data['carbon_credits_potential'] / enhanced_data['area_hectares']
        ).fillna(0)
        
        # Sustainability rating
        conditions = [
            enhanced_data['sustainability_score'] >= 90,
            enhanced_data['sustainability_score'] >= 80,
            enhanced_data['sustainability_score'] >= 70,
            enhanced_data['sustainability_score'] >= 60
        ]
        choices = ['Excellent', 'Good', 'Fair', 'Poor']
        enhanced_data['sustainability_rating'] = np.select(conditions, choices, default='Critical')
        
        logger.info("Calculated derived metrics")
        return enhanced_data
    
    def generate_data_summary(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive data summary statistics"""
        summary = {
            'basic_stats': {},
            'data_quality': {},
            'distributions': {},
            'correlations': {},
            'time_series_stats': {}
        }
        
        # Basic statistics
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        summary['basic_stats'] = data[numerical_columns].describe().to_dict()
        
        # Data quality metrics
        summary['data_quality'] = {
            'total_records': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_records': data.duplicated().sum(),
            'unique_farms': data['farm_id'].nunique() if 'farm_id' in data.columns else 0,
            'unique_states': data['state'].nunique() if 'state' in data.columns else 0,
            'unique_crops': data['crop_type'].nunique() if 'crop_type' in data.columns else 0
        }
        
        # Distribution analysis
        for col in ['co2_sequestered_kg', 'yield_kg_per_ha', 'sustainability_score']:
            if col in data.columns:
                summary['distributions'][col] = {
                    'skewness': float(stats.skew(data[col].dropna())),
                    'kurtosis': float(stats.kurtosis(data[col].dropna())),
                    'quartiles': data[col].quantile([0.25, 0.5, 0.75]).to_dict()
                }
        
        # Correlation analysis
        correlation_cols = ['co2_sequestered_kg', 'yield_kg_per_ha', 'sustainability_score', 
                          'temperature_celsius', 'rainfall_mm']
        available_cols = [col for col in correlation_cols if col in data.columns]
        
        if len(available_cols) > 1:
            corr_matrix = data[available_cols].corr()
            summary['correlations'] = corr_matrix.to_dict()
        
        # Time series statistics (if date column exists)
        if 'date' in data.columns:
            try:
                data['date'] = pd.to_datetime(data['date'])
                summary['time_series_stats'] = {
                    'date_range_start': data['date'].min().isoformat(),
                    'date_range_end': data['date'].max().isoformat(),
                    'total_days': (data['date'].max() - data['date'].min()).days,
                    'records_per_day': len(data) / max(1, (data['date'].max() - data['date'].min()).days)
                }
            except:
                summary['time_series_stats'] = {'error': 'Could not process date column'}
        
        logger.info("Generated comprehensive data summary")
        return summary
    
    def export_processing_report(self, data: pd.DataFrame, output_file: str = None) -> Dict:
        """Export comprehensive data processing report"""
        
        report = {
            'processing_timestamp': datetime.now().isoformat(),
            'data_summary': self.generate_data_summary(data),
            'validation_results': self.validate_data(data),
            'processing_stats': self.processed_stats.copy() if self.processed_stats else {},
            'recommendations': []
        }
        
        # Generate processing recommendations
        quality_score = report['validation_results']['quality_score']
        if quality_score < 85:
            report['recommendations'].append("Consider implementing stricter data collection protocols")
        
        if len(report['validation_results']['anomalies_detected']) > 0:
            report['recommendations'].append("Review anomalous data points for accuracy")
        
        # Save report if filename provided
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Processing report saved to {output_file}")
        
        return report

class DataQualityMonitor:
    """Real-time data quality monitoring for incoming MRV data"""
    
    def __init__(self, quality_thresholds: Dict = None):
        self.quality_thresholds = quality_thresholds or {
            'minimum_quality_score': 80,
            'maximum_anomaly_rate': 0.05,
            'required_fields_completeness': 0.95
        }
        self.alerts = []
    
    def monitor_batch(self, data: pd.DataFrame) -> Dict:
        """Monitor a batch of incoming data for quality issues"""
        
        processor = MRVDataProcessor()
        validation_results = processor.validate_data(data)
        
        monitoring_results = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': len(data),
            'quality_status': 'PASS',
            'alerts': [],
            'metrics': {
                'quality_score': validation_results['quality_score'],
                'anomaly_rate': len(validation_results['anomalies_detected']) / len(data),
                'completeness_rate': validation_results['valid_records'] / len(data)
            }
        }
        
        # Check quality thresholds
        if validation_results['quality_score'] < self.quality_thresholds['minimum_quality_score']:
            monitoring_results['quality_status'] = 'FAIL'
            monitoring_results['alerts'].append({
                'type': 'LOW_QUALITY',
                'message': f"Quality score ({validation_results['quality_score']}%) below threshold"
            })
        
        anomaly_rate = len(validation_results['anomalies_detected']) / len(data)
        if anomaly_rate > self.quality_thresholds['maximum_anomaly_rate']:
            monitoring_results['quality_status'] = 'WARNING'
            monitoring_results['alerts'].append({
                'type': 'HIGH_ANOMALY_RATE',
                'message': f"Anomaly rate ({anomaly_rate:.2%}) above threshold"
            })
        
        completeness_rate = validation_results['valid_records'] / len(data)
        if completeness_rate < self.quality_thresholds['required_fields_completeness']:
            monitoring_results['quality_status'] = 'WARNING'
            monitoring_results['alerts'].append({
                'type': 'LOW_COMPLETENESS',
                'message': f"Data completeness ({completeness_rate:.2%}) below threshold"
            })
        
        # Store alerts for reporting
        self.alerts.extend(monitoring_results['alerts'])
        
        return monitoring_results

if __name__ == "__main__":
    # Demo data processing
    from .generator import MRVDataGenerator
    
    # Generate sample data
    generator = MRVDataGenerator()
    sample_data = generator.generate_comprehensive_dataset(30)
    
    # Initialize processor
    processor = MRVDataProcessor()
    
    # Validate data
    validation_results = processor.validate_data(sample_data)
    print(f"Data Quality Score: {validation_results['quality_score']}%")
    
    # Clean data
    cleaned_data = processor.clean_data(sample_data, 'moderate')
    print(f"Records: {len(sample_data)} -> {len(cleaned_data)}")
    
    # Calculate derived metrics
    enhanced_data = processor.calculate_derived_metrics(cleaned_data)
    print(f"Enhanced data shape: {enhanced_data.shape}")
    
    # Generate summary
    summary = processor.generate_data_summary(enhanced_data)
    print(f"Summary generated with {len(summary.keys())} sections")