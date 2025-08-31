"""
AI/ML Models for Agricultural MRV System
5 specialized models using scikit-learn for predictions and verification
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarbonSequestrationModel:
    """AI Model for predicting carbon sequestration potential"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for carbon sequestration prediction"""
        features = [
            'temperature_celsius', 'humidity_percent', 'rainfall_mm',
            'soil_ph', 'soil_organic_carbon_percent', 'soil_nitrogen_kg_per_ha',
            'area_hectares', 'fertilizer_n_kg', 'water_usage_liters'
        ]
        
        return data[features].values
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the carbon sequestration model"""
        try:
            X = self.prepare_features(data)
            y = data['co2_sequestered_kg'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            feature_names = [
                'Temperature', 'Humidity', 'Rainfall', 'Soil pH',
                'Soil Organic Carbon', 'Soil Nitrogen', 'Area',
                'Fertilizer N', 'Water Usage'
            ]
            self.feature_importance = dict(zip(
                feature_names, self.model.feature_importances_
            ))
            
            self.is_trained = True
            
            metrics = {
                'model_type': 'Carbon Sequestration Prediction',
                'mse': round(mse, 3),
                'r2_score': round(r2, 3),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            logger.info(f"Carbon Sequestration Model trained - R²: {r2:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training carbon sequestration model: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict carbon sequestration"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        X = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        return np.maximum(0, predictions)  # Ensure non-negative

class GHGEmissionsModel:
    """AI Model for predicting GHG emissions"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for GHG emissions prediction"""
        features = [
            'fertilizer_n_kg', 'fertilizer_p_kg', 'pesticide_kg',
            'area_hectares', 'temperature_celsius', 'water_usage_liters',
            'soil_organic_carbon_percent'
        ]
        
        return data[features].values
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the GHG emissions model"""
        try:
            X = self.prepare_features(data)
            
            # Calculate total GHG emissions (CO2 equivalent)
            y = (data['co2_emissions_kg'] + 
                 data['n2o_emissions_kg'] + 
                 data['ch4_emissions_kg']).values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.is_trained = True
            
            metrics = {
                'model_type': 'GHG Emissions Prediction',
                'mse': round(mse, 3),
                'r2_score': round(r2, 3),
                'training_samples': len(X_train)
            }
            
            logger.info(f"GHG Emissions Model trained - R²: {r2:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training GHG emissions model: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict total GHG emissions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        X = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        return np.maximum(0, self.model.predict(X_scaled))

class DataVerificationModel:
    """AI Model for data verification and anomaly detection"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection"""
        features = [
            'temperature_celsius', 'humidity_percent', 'rainfall_mm',
            'soil_ph', 'soil_organic_carbon_percent', 'fertilizer_n_kg',
            'yield_kg_per_ha', 'water_usage_liters', 'sustainability_score'
        ]
        
        return data[features].values
    
    def create_anomaly_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Create anomaly labels based on statistical thresholds"""
        # Define reasonable ranges for different parameters
        anomaly_conditions = [
            (data['temperature_celsius'] < 10) | (data['temperature_celsius'] > 50),
            (data['humidity_percent'] < 20) | (data['humidity_percent'] > 100),
            (data['soil_ph'] < 4.0) | (data['soil_ph'] > 9.0),
            (data['yield_kg_per_ha'] < 100) | (data['yield_kg_per_ha'] > 100000),
            (data['sustainability_score'] < 10) | (data['sustainability_score'] > 100)
        ]
        
        # Mark as anomaly if any condition is true
        anomaly_mask = np.any(anomaly_conditions, axis=0)
        
        # Add some random normal variations as valid data
        normal_samples = int(len(data) * 0.85)  # 85% normal data
        labels = np.zeros(len(data), dtype=int)
        labels[anomaly_mask] = 1  # 1 = anomaly
        
        return labels
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the data verification model"""
        try:
            X = self.prepare_features(data)
            y = self.create_anomaly_labels(data)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            
            metrics = {
                'model_type': 'Data Verification & Anomaly Detection',
                'accuracy': round(accuracy, 3),
                'training_samples': len(X_train),
                'anomaly_rate': round(np.mean(y_train), 3)
            }
            
            logger.info(f"Data Verification Model trained - Accuracy: {accuracy:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training data verification model: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict data validity and confidence scores"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        X = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Return predictions (0=valid, 1=anomaly) and confidence scores
        confidence = np.max(probabilities, axis=1)
        
        return predictions, confidence

class CropYieldModel:
    """AI Model for crop yield prediction"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.crop_encoder = LabelEncoder()
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for yield prediction"""
        # Encode crop types
        crop_encoded = self.crop_encoder.fit_transform(data['crop_type'])
        
        numerical_features = data[[
            'temperature_celsius', 'humidity_percent', 'rainfall_mm',
            'soil_ph', 'soil_organic_carbon_percent', 'soil_nitrogen_kg_per_ha',
            'fertilizer_n_kg', 'water_usage_liters', 'area_hectares'
        ]].values
        
        # Combine encoded crops with numerical features
        features = np.column_stack([crop_encoded, numerical_features])
        
        return features
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the crop yield model"""
        try:
            X = self.prepare_features(data)
            y = data['yield_kg_per_ha'].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.is_trained = True
            
            metrics = {
                'model_type': 'Crop Yield Prediction',
                'mse': round(mse, 3),
                'r2_score': round(r2, 3),
                'training_samples': len(X_train)
            }
            
            logger.info(f"Crop Yield Model trained - R²: {r2:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training crop yield model: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict crop yield"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        X = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        return np.maximum(0, self.model.predict(X_scaled))

class WaterOptimizationModel:
    """AI Model for water usage optimization"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for water optimization"""
        features = [
            'temperature_celsius', 'humidity_percent', 'rainfall_mm',
            'area_hectares', 'yield_kg_per_ha', 'soil_organic_carbon_percent'
        ]
        
        return data[features].values
    
    def create_optimal_water_target(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate optimal water usage based on efficiency"""
        # Optimal water = f(yield, area, climate conditions)
        base_water = data['area_hectares'] * 50  # Base 50L per hectare
        
        # Climate adjustments
        temp_factor = np.maximum(0.5, (data['temperature_celsius'] - 20) / 15)
        humidity_factor = np.maximum(0.3, (100 - data['humidity_percent']) / 80)
        rain_factor = 1 - np.minimum(0.8, data['rainfall_mm'] / 20)
        
        optimal_water = base_water * temp_factor * humidity_factor * rain_factor
        
        return optimal_water
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the water optimization model"""
        try:
            X = self.prepare_features(data)
            y = self.create_optimal_water_target(data)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.is_trained = True
            
            metrics = {
                'model_type': 'Water Usage Optimization',
                'mse': round(mse, 3),
                'r2_score': round(r2, 3),
                'training_samples': len(X_train)
            }
            
            logger.info(f"Water Optimization Model trained - R²: {r2:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training water optimization model: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict optimal water usage"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        X = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        return np.maximum(0, self.model.predict(X_scaled))

class AIModelManager:
    """Manager for all AI/ML models"""
    
    def __init__(self):
        self.models = {
            'carbon_sequestration': CarbonSequestrationModel(),
            'ghg_emissions': GHGEmissionsModel(),
            'data_verification': DataVerificationModel(),
            'crop_yield': CropYieldModel(),
            'water_optimization': WaterOptimizationModel()
        }
        self.training_metrics = {}
        
    def train_all_models(self, data: pd.DataFrame) -> Dict:
        """Train all AI models with provided data"""
        logger.info("Starting training of all AI models...")
        
        all_metrics = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name} model...")
                metrics = model.train(data)
                all_metrics[model_name] = metrics
                self.training_metrics[model_name] = metrics
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                all_metrics[model_name] = {'error': str(e)}
        
        logger.info("All models training completed!")
        return all_metrics
    
    def get_all_predictions(self, data: pd.DataFrame) -> Dict:
        """Get predictions from all trained models"""
        predictions = {}
        
        try:
            # Carbon sequestration
            if self.models['carbon_sequestration'].is_trained:
                predictions['carbon_sequestration'] = self.models['carbon_sequestration'].predict(data)
            
            # GHG emissions
            if self.models['ghg_emissions'].is_trained:
                predictions['ghg_emissions'] = self.models['ghg_emissions'].predict(data)
            
            # Data verification
            if self.models['data_verification'].is_trained:
                validity, confidence = self.models['data_verification'].predict(data)
                predictions['data_validity'] = validity
                predictions['confidence_scores'] = confidence
            
            # Crop yield
            if self.models['crop_yield'].is_trained:
                predictions['crop_yield'] = self.models['crop_yield'].predict(data)
            
            # Water optimization
            if self.models['water_optimization'].is_trained:
                predictions['optimal_water'] = self.models['water_optimization'].predict(data)
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            predictions['error'] = str(e)
        
        return predictions
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for all trained models"""
        return self.training_metrics.copy()
    
    def save_models(self, filepath: str) -> None:
        """Save all trained models to file"""
        try:
            model_data = {
                'models': self.models,
                'metrics': self.training_metrics
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, filepath: str) -> None:
        """Load trained models from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.training_metrics = model_data['metrics']
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

def demo_ai_models():
    """Demonstrate AI models functionality"""
    from .mrv_node import create_demo_farms
    
    # Create sample data
    farms = create_demo_farms()
    all_data = []
    
    for farm in farms[:5]:  # Use first 5 farms
        data = farm.generate_historical_data(30)
        all_data.append(data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Initialize and train models
    ai_manager = AIModelManager()
    training_results = ai_manager.train_all_models(combined_data)
    
    # Get predictions on sample data
    sample_predictions = ai_manager.get_all_predictions(combined_data.head(10))
    
    print("=== AI Models Demo Results ===")
    for model_name, metrics in training_results.items():
        if 'error' not in metrics:
            print(f"{model_name}: R²={metrics.get('r2_score', metrics.get('accuracy', 'N/A'))}")
    
    return ai_manager, training_results

if __name__ == "__main__":
    demo_ai_models()