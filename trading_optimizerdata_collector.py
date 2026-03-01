"""
Market data collector with CCXT integration.
Handles real-time and historical data with robust error handling.
"""
import pandas as pd
import numpy as np
import ccxt
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time

class DataCollector:
    """
    Robust market data collector with rate limiting, retry logic,
    and multiple exchange support via CCXT.
    """
    
    def __init__(self, exchange_id: str = "binance", config: Optional[Dict] = None):
        """
        Initialize data collector with exchange configuration.
        
        Args:
            exchange_id: CCXT exchange identifier
            config: Exchange configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.exchange_id = exchange_id
        self.config = config or {}
        self.exchange = None
        self._initialize_exchange()
        self.request_count = 0
        self.last_request_time = datetime.now()
        
    def _initialize_exchange(self) -> None:
        """Initialize CCXT exchange with proper error handling"""
        try:
            # Get exchange class from CCXT
            exchange_class = getattr(ccxt, self.exchange_id)
            
            # Configure exchange with error handling
            exchange_config = {
                'enableRateLimit': self.config.get('enable_rate_limit', True),
                'options': {'defaultType': 'spot'},
                'timeout': 30000,
            }
            
            # Add credentials if provided
            api_key = self.config.get('apiKey')
            api_secret = self.config.get('secret')
            if api_key and api_secret:
                exchange_config['apiKey'] = api_key
                exchange_config['secret'] = api_secret
            
            # Create exchange instance
            self.exchange = exchange_class(exchange_config)
            self.logger.info(f"Initialized {self