import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Tuple
import pandas as pd

class FraudDetectionService:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self._train_mock_model()
    
    def _train_mock_model(self):
        """Train fraud detection model with mock transaction data"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate normal transactions
        normal_data = {
            'amount': np.random.lognormal(8, 1.5, int(n_samples * 0.9)),  # Most transactions small
            'hour': np.random.choice(range(8, 22), int(n_samples * 0.9)),  # Business hours
            'location_risk': np.random.uniform(0, 0.3, int(n_samples * 0.9)),  # Low risk locations
            'merchant_risk': np.random.uniform(0, 0.2, int(n_samples * 0.9))   # Trusted merchants
        }
        
        # Generate fraudulent transactions
        fraud_data = {
            'amount': np.random.uniform(50000, 200000, int(n_samples * 0.1)),  # Large amounts
            'hour': np.random.choice(range(0, 6), int(n_samples * 0.1)),      # Odd hours
            'location_risk': np.random.uniform(0.7, 1.0, int(n_samples * 0.1)),  # High risk
            'merchant_risk': np.random.uniform(0.8, 1.0, int(n_samples * 0.1))    # Suspicious merchants
        }
        
        # Combine data
        all_data = {}
        for key in normal_data:
            all_data[key] = np.concatenate([normal_data[key], fraud_data[key]])
        
        df = pd.DataFrame(all_data)
        self.model.fit(df)
        
        # Store feature names for consistency
        self.features = ['amount', 'hour', 'location_risk', 'merchant_risk']
    
    def detect_fraud(self, transaction_data: Dict[str, Any]) -> Tuple[float, bool]:
        """Detect fraud using business rules and anomaly detection"""
        
        # Extract features
        amount = transaction_data.get('amount', 0)
        hour = transaction_data.get('time_of_day', 12)
        location = transaction_data.get('location', '').lower()
        merchant = transaction_data.get('merchant', '').lower()
        
        # Calculate risk scores
        amount_risk = self._calculate_amount_risk(amount)
        time_risk = self._calculate_time_risk(hour)
        location_risk = self._calculate_location_risk(location)
        merchant_risk = self._calculate_merchant_risk(merchant)
        
        # Prepare data for model
        features = np.array([[amount, hour, location_risk, merchant_risk]])
        
        # Get anomaly score (-1 for anomaly, 1 for normal)
        anomaly_score = self.model.decision_function(features)[0]
        
        # Calculate combined fraud probability
        rule_based_risk = (amount_risk + time_risk + location_risk + merchant_risk) / 4
        
        # Normalize anomaly score to probability (higher negative score = more anomalous)
        anomaly_prob = max(0, (0.5 - anomaly_score) / 1.0)
        
        # Combine both approaches
        fraud_probability = (rule_based_risk * 0.6 + anomaly_prob * 0.4)
        fraud_probability = min(1.0, max(0.0, fraud_probability))
        
        # Threshold for fraud detection
        is_fraud = fraud_probability > 0.7
        
        return fraud_probability, is_fraud
    
    def _calculate_amount_risk(self, amount: float) -> float:
        """Calculate risk based on transaction amount"""
        if amount > 100000:
            return 0.9
        elif amount > 50000:
            return 0.6
        elif amount > 10000:
            return 0.3
        else:
            return 0.1
    
    def _calculate_time_risk(self, hour: int) -> float:
        """Calculate risk based on time of transaction"""
        if 0 <= hour <= 5 or 23 <= hour <= 23:
            return 0.8  # Very early morning or late night
        elif 6 <= hour <= 8 or 22 <= hour <= 22:
            return 0.4  # Early morning or late evening
        else:
            return 0.1  # Normal business hours
    
    def _calculate_location_risk(self, location: str) -> float:
        """Calculate risk based on location (simplified)"""
        high_risk_keywords = ['unknown', 'foreign', 'offshore', 'anonymous']
        medium_risk_keywords = ['atm', 'gas', 'convenience']
        
        location_lower = location.lower()
        
        for keyword in high_risk_keywords:
            if keyword in location_lower:
                return 0.9
        
        for keyword in medium_risk_keywords:
            if keyword in location_lower:
                return 0.4
        
        return 0.1  # Default low risk
    
    def _calculate_merchant_risk(self, merchant: str) -> float:
        """Calculate risk based on merchant (simplified)"""
        high_risk_keywords = ['casino', 'gambling', 'crypto', 'bitcoin', 'offshore']
        medium_risk_keywords = ['cash', 'advance', 'pawn', 'gold']
        
        merchant_lower = merchant.lower()
        
        for keyword in high_risk_keywords:
            if keyword in merchant_lower:
                return 0.9
        
        for keyword in medium_risk_keywords:
            if keyword in merchant_lower:
                return 0.5
        
        return 0.1  # Default low risk
