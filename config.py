"""
Configuration Settings for AgroMRV System
Central configuration management for the agricultural MRV application
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Base directories
PROJECT_ROOT = Path(__file__).parent
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_DIR = PROJECT_ROOT / "exports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, EXPORTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

class AgroMRVConfig:
    """Main configuration class for AgroMRV system"""
    
    # Application Information
    APP_NAME = "AgroMRV System"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "AI-Powered Agricultural Monitoring, Reporting & Verification"
    
    # NABARD Hackathon Information
    NABARD_HACKATHON = "NABARD Hackathon 2025"
    NABARD_THEME = "Climate Finance for Smallholder Farmers"
    TARGET_AUDIENCE = "Indian Smallholder Farmers"
    
    # System Configuration
    DEBUG_MODE = os.getenv('DEBUG', 'True').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Database Configuration (for future extension)
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///agromrv.db')
    
    # AI/ML Configuration
    ML_MODEL_CONFIG = {
        'n_estimators': 100,
        'max_depth': 12,
        'random_state': 42,
        'n_jobs': -1,
        'test_size': 0.2,
        'cross_validation_folds': 5
    }
    
    # Blockchain Configuration
    BLOCKCHAIN_CONFIG = {
        'mining_difficulty': 3,
        'mining_reward': 1.0,
        'block_size_limit': 1000000,  # 1MB
        'transaction_fee': 0.0001
    }
    
    # IPCC Compliance Configuration
    IPCC_CONFIG = {
        'methodology_tier': 'Tier 2',
        'guidelines_version': '2019 Refinement to 2006 IPCC Guidelines',
        'gwp_timeframe': 100,  # 100-year GWP values
        'uncertainty_level': 'Medium',
        'geographic_scope': 'India',
        'sectoral_scope': 'Agriculture, Forestry and Other Land Use (AFOLU)'
    }
    
    # Data Generation Configuration
    DATA_CONFIG = {
        'default_data_days': 60,
        'max_data_days': 365,
        'default_farms_count': 15,
        'min_area_hectares': 0.1,
        'max_area_hectares': 50.0,
        'data_quality_threshold': 0.95
    }
    
    # Dashboard Configuration
    DASHBOARD_CONFIG = {
        'page_title': 'AgroMRV System',
        'page_icon': '🌱',
        'layout': 'wide',
        'sidebar_state': 'expanded',
        'theme': 'light',
        'max_table_rows': 1000,
        'chart_height': 500
    }
    
    # Export Configuration
    EXPORT_CONFIG = {
        'default_format': 'JSON',
        'include_metadata': True,
        'compression': True,
        'max_file_size_mb': 100,
        'supported_formats': ['CSV', 'JSON', 'PDF', 'Excel']
    }
    
    # Indian Agricultural Configuration
    INDIAN_STATES = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
        'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
        'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
        'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
        'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
    ]
    
    # Priority states for the application
    PRIORITY_STATES = [
        'Punjab', 'Haryana', 'Uttar Pradesh', 'Bihar', 'West Bengal',
        'Madhya Pradesh', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Kerala',
        'Andhra Pradesh', 'Telangana', 'Gujarat', 'Rajasthan', 'Odisha'
    ]
    
    # Supported crop types
    CROP_TYPES = {
        'rice': {
            'scientific_name': 'Oryza sativa',
            'season': 'kharif',
            'water_requirement': 'high',
            'typical_yield_kg_per_ha': 4000,
            'carbon_intensity': 'medium'
        },
        'wheat': {
            'scientific_name': 'Triticum aestivum',
            'season': 'rabi',
            'water_requirement': 'medium',
            'typical_yield_kg_per_ha': 3200,
            'carbon_intensity': 'low'
        },
        'maize': {
            'scientific_name': 'Zea mays',
            'season': 'kharif',
            'water_requirement': 'medium',
            'typical_yield_kg_per_ha': 3800,
            'carbon_intensity': 'low'
        },
        'vegetables': {
            'scientific_name': 'Various',
            'season': 'both',
            'water_requirement': 'high',
            'typical_yield_kg_per_ha': 15000,
            'carbon_intensity': 'medium'
        },
        'millet': {
            'scientific_name': 'Pennisetum glaucum',
            'season': 'kharif',
            'water_requirement': 'low',
            'typical_yield_kg_per_ha': 1800,
            'carbon_intensity': 'low'
        },
        'sugarcane': {
            'scientific_name': 'Saccharum officinarum',
            'season': 'annual',
            'water_requirement': 'very_high',
            'typical_yield_kg_per_ha': 70000,
            'carbon_intensity': 'high'
        },
        'cotton': {
            'scientific_name': 'Gossypium hirsutum',
            'season': 'kharif',
            'water_requirement': 'high',
            'typical_yield_kg_per_ha': 500,
            'carbon_intensity': 'medium'
        },
        'pulses': {
            'scientific_name': 'Various legumes',
            'season': 'both',
            'water_requirement': 'low',
            'typical_yield_kg_per_ha': 1200,
            'carbon_intensity': 'negative'  # N-fixing crops
        }
    }
    
    # Climate zones for India
    CLIMATE_ZONES = {
        'tropical_wet': ['Kerala', 'Karnataka', 'Tamil Nadu', 'Andhra Pradesh'],
        'tropical_dry': ['Rajasthan', 'Gujarat', 'Maharashtra'],
        'subtropical_humid': ['West Bengal', 'Bihar', 'Uttar Pradesh'],
        'subtropical_dry': ['Punjab', 'Haryana', 'Delhi'],
        'mountain': ['Himachal Pradesh', 'Uttarakhand', 'Jammu and Kashmir']
    }
    
    # API Configuration (for future extension)
    API_CONFIG = {
        'version': 'v1',
        'base_url': '/api/v1',
        'rate_limit_per_minute': 100,
        'authentication': 'api_key',
        'cors_enabled': True,
        'swagger_enabled': True
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': LOG_LEVEL,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': str(LOGS_DIR / f'agromrv_{datetime.now().strftime("%Y%m%d")}.log'),
                'mode': 'a'
            }
        },
        'loggers': {
            'agromrv': {
                'level': LOG_LEVEL,
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': LOG_LEVEL,
            'handlers': ['console']
        }
    }
    
    # Security Configuration
    SECURITY_CONFIG = {
        'secret_key': os.getenv('SECRET_KEY', 'agromrv-secret-key-change-in-production'),
        'password_min_length': 8,
        'session_timeout_minutes': 30,
        'max_login_attempts': 5,
        'rate_limit_enabled': True
    }
    
    # Performance Configuration
    PERFORMANCE_CONFIG = {
        'cache_enabled': True,
        'cache_timeout_seconds': 300,
        'max_concurrent_requests': 100,
        'connection_pool_size': 20,
        'query_timeout_seconds': 30
    }
    
    # Monitoring Configuration
    MONITORING_CONFIG = {
        'health_check_enabled': True,
        'metrics_collection': True,
        'error_reporting': True,
        'performance_tracking': True
    }
    
    # Feature Flags
    FEATURE_FLAGS = {
        'ai_predictions_enabled': True,
        'blockchain_verification_enabled': True,
        'ipcc_compliance_enabled': True,
        'export_functionality_enabled': True,
        'demo_mode_available': True,
        'batch_processing_enabled': True,
        'real_time_updates_enabled': False  # For future enhancement
    }

class EnvironmentConfig:
    """Environment-specific configurations"""
    
    @staticmethod
    def get_config(environment: str = None) -> Dict:
        """Get configuration based on environment"""
        
        env = environment or os.getenv('ENVIRONMENT', 'development')
        
        base_config = {
            'app_name': AgroMRVConfig.APP_NAME,
            'debug': AgroMRVConfig.DEBUG_MODE,
            'log_level': AgroMRVConfig.LOG_LEVEL
        }
        
        if env == 'development':
            return {
                **base_config,
                'debug': True,
                'log_level': 'DEBUG',
                'database_url': 'sqlite:///dev_agromrv.db',
                'cache_enabled': False,
                'blockchain_difficulty': 2  # Lower for faster development
            }
        
        elif env == 'testing':
            return {
                **base_config,
                'debug': False,
                'log_level': 'WARNING',
                'database_url': 'sqlite:///test_agromrv.db',
                'cache_enabled': False,
                'blockchain_difficulty': 1
            }
        
        elif env == 'production':
            return {
                **base_config,
                'debug': False,
                'log_level': 'ERROR',
                'database_url': os.getenv('DATABASE_URL'),
                'cache_enabled': True,
                'blockchain_difficulty': 4,
                'security_enhanced': True
            }
        
        else:
            return base_config

class ValidationConfig:
    """Data validation configuration"""
    
    FARM_VALIDATION = {
        'farm_id': {
            'required': True,
            'max_length': 20,
            'pattern': r'^[A-Z0-9_]+$'
        },
        'area_hectares': {
            'required': True,
            'min_value': 0.1,
            'max_value': 50.0,
            'data_type': 'float'
        },
        'state': {
            'required': True,
            'allowed_values': AgroMRVConfig.INDIAN_STATES
        },
        'crop_type': {
            'required': True,
            'allowed_values': list(AgroMRVConfig.CROP_TYPES.keys())
        }
    }
    
    ENVIRONMENTAL_VALIDATION = {
        'temperature_celsius': {
            'min_value': -10.0,
            'max_value': 55.0,
            'data_type': 'float'
        },
        'humidity_percent': {
            'min_value': 10.0,
            'max_value': 100.0,
            'data_type': 'float'
        },
        'rainfall_mm': {
            'min_value': 0.0,
            'max_value': 500.0,
            'data_type': 'float'
        },
        'soil_ph': {
            'min_value': 3.0,
            'max_value': 10.0,
            'data_type': 'float'
        }
    }

def setup_logging():
    """Setup logging configuration"""
    import logging.config
    logging.config.dictConfig(AgroMRVConfig.LOGGING_CONFIG)

def get_database_url(environment: str = None) -> str:
    """Get database URL for the specified environment"""
    env_config = EnvironmentConfig.get_config(environment)
    return env_config.get('database_url', AgroMRVConfig.DATABASE_URL)

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled"""
    return AgroMRVConfig.FEATURE_FLAGS.get(feature_name, False)

def get_crop_info(crop_type: str) -> Dict:
    """Get detailed information about a crop type"""
    return AgroMRVConfig.CROP_TYPES.get(crop_type, {})

def get_state_climate_zone(state: str) -> str:
    """Get climate zone for a given state"""
    for zone, states in AgroMRVConfig.CLIMATE_ZONES.items():
        if state in states:
            return zone
    return 'unknown'

def validate_configuration():
    """Validate system configuration"""
    errors = []
    
    # Check required directories
    for directory in [DATA_DIR, EXPORTS_DIR, LOGS_DIR]:
        if not directory.exists():
            errors.append(f"Required directory missing: {directory}")
    
    # Check feature dependencies
    if AgroMRVConfig.FEATURE_FLAGS['blockchain_verification_enabled']:
        if not AgroMRVConfig.FEATURE_FLAGS['ai_predictions_enabled']:
            errors.append("Blockchain verification requires AI predictions to be enabled")
    
    # Check environment variables for production
    if os.getenv('ENVIRONMENT') == 'production':
        required_env_vars = ['DATABASE_URL', 'SECRET_KEY']
        for var in required_env_vars:
            if not os.getenv(var):
                errors.append(f"Required environment variable missing: {var}")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return True

if __name__ == "__main__":
    # Configuration validation and testing
    print(f"🌱 {AgroMRVConfig.APP_NAME} v{AgroMRVConfig.APP_VERSION}")
    print(f"📋 Configuration Status:")
    
    try:
        validate_configuration()
        print("✅ Configuration validation passed")
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
    
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"🗂️  Data directory: {DATA_DIR}")
    print(f"📤 Exports directory: {EXPORTS_DIR}")
    print(f"📝 Logs directory: {LOGS_DIR}")
    print(f"🚩 Feature flags: {sum(AgroMRVConfig.FEATURE_FLAGS.values())}/{len(AgroMRVConfig.FEATURE_FLAGS)} enabled")
    print(f"🌾 Supported crops: {len(AgroMRVConfig.CROP_TYPES)}")
    print(f"🗺️  Priority states: {len(AgroMRVConfig.PRIORITY_STATES)}")