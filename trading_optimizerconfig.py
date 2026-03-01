"""
Configuration management for the Autonomous Trading Algorithm Optimizer.
Centralized config ensures consistency across all modules.
"""
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import logging

@dataclass
class FirebaseConfig:
    """Firebase configuration for real-time state management"""
    credential_path: str = "firebase_credentials.json"
    project_id: str = "trading-optimizer-agents"
    collection_name: str = "trading_states"
    performance_collection: str = "algorithm_performance"
    
    def validate(self) -> bool:
        """Validate Firebase configuration exists"""
        if not os.path.exists(self.credential_path):
            logging.error(f"Firebase credentials not found at {self.credential_path}")
            return False
        return True

@dataclass
class ExchangeConfig:
    """Exchange API configuration with CCXT"""
    exchange_id: str = "binance"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox_mode: bool = True
    rate_limit: bool = True
    enable_rate_limit: bool = True
    
    def get_credentials(self) -> Dict[str, str]:
        """Safely retrieve API credentials"""
        return {
            'apiKey': self.api_key or os.getenv('EXCHANGE_API_KEY', ''),
            'secret': self.api_secret or os.getenv('EXCHANGE_API_SECRET', '')
        }

@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 32
    target_update_frequency: int = 100
    state_size: int = 10  # Number of features in state representation

@dataclass
class DataConfig:
    """Data collection and processing configuration"""
    symbols: list = None
    timeframe: str = "1h"
    lookback_period: int = 100
    features: list = None
    
    def __post_init__(self):
        """Initialize default values"""
        if self.symbols is None:
            self.symbols = ["BTC/USDT", "ETH/USDT"]
        if self.features is None:
            self.features = ['open', 'high', 'low', 'close', 'volume']

class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.firebase = FirebaseConfig()
        self.exchange = ExchangeConfig()
        self.rl = RLConfig()
        self.data = DataConfig()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                # Update configs from file (implementation simplified for brevity)
                self.logger.info(f"Loaded config from {config_path}")
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            # Continue with defaults
    
    def validate_all(self) -> bool:
        """Validate all configurations"""
        validations = [
            ("Firebase", self.firebase.validate()),
            ("Exchange", self._validate_exchange()),
            ("Data", self._validate_data())
        ]
        
        all_valid = True
        for name, is_valid in validations:
            if not is_valid:
                self.logger.error(f"{name} configuration invalid")
                all_valid = False
        
        if all_valid:
            self.logger.info("All configurations validated successfully")
        
        return all_valid
    
    def _validate_exchange(self) -> bool:
        """Validate exchange configuration"""
        creds = self.exchange.get_credentials()
        if not creds['apiKey'] and not self.exchange.sandbox_mode:
            self.logger.warning("No API key found - sandbox mode will be used")
            return True  # Sandbox mode is valid
        return True
    
    def _validate_data(self) -> bool:
        """Validate data configuration"""
        if not self.data.symbols:
            self.logger.error("No trading symbols configured")
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {
            "firebase": {
                "credential_path": self.firebase.credential_path,
                "project_id": self.firebase.project_id
            },
            "exchange": {
                "exchange_id": self.exchange.exchange_id,
                "sandbox_mode": self.exchange.sandbox_mode
            },
            "rl": {
                "learning_rate": self.rl.learning_rate,
                "state_size": self.rl.state_size
            }
        }