import os
import logging
import asyncio
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from .gemini_ai_service import GeminiAIService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFraudDetectionService:
    def __init__(self):
        # Initialize AI service for advanced analysis
        self.ai_service = GeminiAIService()
        
        # Initialize anomaly detection model
        self.anomaly_model = IsolationForest(contamination=0.15, random_state=42)
        self.scaler = StandardScaler()
        
        # Fraud patterns and rules
        self.fraud_patterns = {
            'velocity_fraud': {
                'max_transactions_per_hour': 10,
                'max_amount_per_hour': 500000,
                'suspicious_time_pattern': [0, 1, 2, 3, 4, 5, 23]  # Late night/early morning
            },
            'amount_fraud': {
                'round_amounts': [10000, 25000, 50000, 100000, 500000],
                'high_risk_threshold': 100000,
                'micro_transaction_threshold': 10
            },
            'behavioral_fraud': {
                'new_account_threshold_days': 30,
                'dormant_account_threshold_days': 180,
                'unusual_location_radius_km': 500
            },
            'merchant_fraud': {
                'high_risk_categories': ['gambling', 'casino', 'crypto', 'offshore', 'cash_advance'],
                'suspicious_merchants': ['unknown', 'anonymous', 'temp', 'test']
            }
        }
        
        # Risk scoring weights
        self.risk_weights = {
            'amount_risk': 0.25,
            'velocity_risk': 0.30,
            'behavioral_risk': 0.20,
            'merchant_risk': 0.15,
            'ai_risk': 0.10
        }
        
        # Train the anomaly detection model with synthetic data
        self._train_anomaly_model()
        
        logger.info("Enhanced fraud detection service initialized")
    
    def _train_anomaly_model(self):
        """Train anomaly detection model with synthetic transaction data"""
        try:
            # Generate synthetic normal transaction data
            np.random.seed(42)
            n_normal = 1000
            
            normal_data = {
                'amount': np.random.lognormal(6, 1.5, n_normal),  # Normal transaction amounts
                'hour': np.random.choice(range(6, 23), n_normal, p=[0.08]*17),  # Business hours
                'day_of_week': np.random.choice(range(7), n_normal),
                'merchant_risk': np.random.beta(2, 5, n_normal),  # Lower risk merchants
                'location_risk': np.random.beta(2, 8, n_normal),  # Lower risk locations
                'user_age_days': np.random.exponential(365, n_normal),
                'velocity_score': np.random.beta(2, 8, n_normal)
            }
            
            df_normal = pd.DataFrame(normal_data)
            
            # Scale and train
            X_scaled = self.scaler.fit_transform(df_normal)
            self.anomaly_model.fit(X_scaled)
            
            logger.info("Anomaly detection model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training anomaly model: {e}")
    
    async def enhanced_fraud_detection(self, transaction_data: Dict[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """Enhanced fraud detection using multiple techniques"""
        try:
            # Extract and analyze transaction features
            features = await self._extract_fraud_features(transaction_data)
            
            # Calculate individual risk scores
            amount_risk = await self._calculate_amount_risk(transaction_data, features)
            velocity_risk = await self._calculate_velocity_risk(transaction_data, features)
            behavioral_risk = await self._calculate_behavioral_risk(transaction_data, features)
            merchant_risk = await self._calculate_merchant_risk(transaction_data, features)
            
            # Get AI-powered risk assessment
            ai_risk = await self._get_ai_fraud_assessment(transaction_data, features)
            
            # Calculate weighted fraud probability
            fraud_probability = (
                amount_risk * self.risk_weights['amount_risk'] +
                velocity_risk * self.risk_weights['velocity_risk'] +
                behavioral_risk * self.risk_weights['behavioral_risk'] +
                merchant_risk * self.risk_weights['merchant_risk'] +
                ai_risk * self.risk_weights['ai_risk']
            )
            
            # Apply anomaly detection
            anomaly_score = await self._get_anomaly_score(features)
            
            # Combine scores
            final_fraud_probability = min(1.0, (fraud_probability * 0.8) + (anomaly_score * 0.2))
            
            # Determine if fraud
            is_fraud = final_fraud_probability > 0.7
            
            # Generate detailed fraud factors
            fraud_factors = await self._generate_fraud_factors(
                transaction_data, features, {
                    'amount_risk': amount_risk,
                    'velocity_risk': velocity_risk,
                    'behavioral_risk': behavioral_risk,
                    'merchant_risk': merchant_risk,
                    'ai_risk': ai_risk,
                    'anomaly_score': anomaly_score
                }
            )
            
            logger.info(f"Fraud detection completed: {final_fraud_probability:.3f} probability, fraud={is_fraud}")
            
            return final_fraud_probability, is_fraud, fraud_factors
            
        except Exception as e:
            logger.error(f"Error in fraud detection: {e}")
            return await self._fallback_fraud_detection(transaction_data)
    
    async def _extract_fraud_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for fraud analysis"""
        features = {
            'amount': transaction_data.get('amount', 0),
            'hour': transaction_data.get('time_of_day', 12),
            'day_of_week': datetime.now().weekday(),
            'merchant': transaction_data.get('merchant', '').lower(),
            'location': transaction_data.get('location', '').lower(),
            'user_id': transaction_data.get('user_id', ''),
            'transaction_id': transaction_data.get('transaction_id', ''),
            'payment_method': transaction_data.get('payment_method', 'card').lower(),
            'user_history': transaction_data.get('user_history', []),
            'account_age': transaction_data.get('account_age', 100)
        }
        
        # Calculate derived features
        features['is_weekend'] = features['day_of_week'] >= 5
        features['is_night'] = features['hour'] < 6 or features['hour'] > 22
        features['amount_rounded'] = features['amount'] % 1000 == 0
        features['transaction_count_24h'] = len([
            tx for tx in features['user_history'] 
            if self._is_within_timeframe(tx.get('timestamp', ''), hours=24)
        ])
        
        return features
    
    async def _calculate_amount_risk(self, transaction_data: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate risk score based on transaction amount"""
        amount = features['amount']
        
        risk_score = 0.0
        
        # High amount risk
        if amount > self.fraud_patterns['amount_fraud']['high_risk_threshold']:
            risk_score += 0.6
        elif amount > 50000:
            risk_score += 0.3
        elif amount > 10000:
            risk_score += 0.1
        
        # Round amount risk (common in fraud)
        if amount in self.fraud_patterns['amount_fraud']['round_amounts']:
            risk_score += 0.2
        elif amount % 1000 == 0 and amount >= 5000:
            risk_score += 0.1
        
        # Micro transaction risk (testing stolen cards)
        if amount <= self.fraud_patterns['amount_fraud']['micro_transaction_threshold']:
            risk_score += 0.4
        
        # Compare with user's typical transaction amounts
        user_history = features.get('user_history', [])
        if user_history:
            avg_amount = np.mean([tx.get('amount', 0) for tx in user_history])
            if avg_amount > 0 and amount > avg_amount * 5:
                risk_score += 0.3  # Amount much higher than usual
        
        return min(1.0, risk_score)
    
    async def _calculate_velocity_risk(self, transaction_data: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate risk score based on transaction velocity"""
        user_history = features.get('user_history', [])
        
        risk_score = 0.0
        
        # Transaction frequency risk
        transactions_1h = len([
            tx for tx in user_history 
            if self._is_within_timeframe(tx.get('timestamp', ''), hours=1)
        ])
        
        transactions_24h = features.get('transaction_count_24h', 0)
        
        # High frequency in short time
        if transactions_1h > self.fraud_patterns['velocity_fraud']['max_transactions_per_hour']:
            risk_score += 0.8
        elif transactions_1h > 5:
            risk_score += 0.4
        elif transactions_1h > 2:
            risk_score += 0.2
        
        # High frequency in 24 hours
        if transactions_24h > 20:
            risk_score += 0.6
        elif transactions_24h > 10:
            risk_score += 0.3
        
        # Amount velocity risk
        amount_1h = sum([
            tx.get('amount', 0) for tx in user_history 
            if self._is_within_timeframe(tx.get('timestamp', ''), hours=1)
        ])
        
        if amount_1h > self.fraud_patterns['velocity_fraud']['max_amount_per_hour']:
            risk_score += 0.7
        
        # Time pattern risk
        if features['hour'] in self.fraud_patterns['velocity_fraud']['suspicious_time_pattern']:
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    async def _calculate_behavioral_risk(self, transaction_data: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate risk score based on behavioral patterns"""
        risk_score = 0.0
        
        account_age = features.get('account_age', 100)
        user_history = features.get('user_history', [])
        
        # New account risk
        if account_age < self.fraud_patterns['behavioral_fraud']['new_account_threshold_days']:
            risk_score += 0.5
            
            # New account with high amount is very suspicious
            if features['amount'] > 10000:
                risk_score += 0.3
        
        # Dormant account suddenly active
        if user_history:
            last_transaction = max([
                datetime.fromisoformat(tx.get('timestamp', '').replace('Z', '+00:00')) 
                for tx in user_history 
                if tx.get('timestamp')
            ], default=datetime.now() - timedelta(days=1))
            
            days_since_last = (datetime.now() - last_transaction).days
            
            if days_since_last > self.fraud_patterns['behavioral_fraud']['dormant_account_threshold_days']:
                risk_score += 0.4
        
        # Unusual transaction pattern
        if features['is_night'] and features['is_weekend']:
            risk_score += 0.2
        elif features['is_night']:
            risk_score += 0.1
        
        # Payment method risk
        payment_method = features.get('payment_method', 'card')
        if payment_method in ['crypto', 'prepaid', 'gift_card']:
            risk_score += 0.3
        elif payment_method in ['wire', 'cash']:
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    async def _calculate_merchant_risk(self, transaction_data: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate risk score based on merchant"""
        merchant = features.get('merchant', '').lower()
        location = features.get('location', '').lower()
        
        risk_score = 0.0
        
        # High-risk merchant categories
        for category in self.fraud_patterns['merchant_fraud']['high_risk_categories']:
            if category in merchant:
                risk_score += 0.6
                break
        
        # Suspicious merchant names
        for suspicious in self.fraud_patterns['merchant_fraud']['suspicious_merchants']:
            if suspicious in merchant:
                risk_score += 0.5
                break
        
        # Location-based risk
        high_risk_locations = ['unknown', 'foreign', 'offshore', 'anonymous']
        for loc in high_risk_locations:
            if loc in location:
                risk_score += 0.4
                break
        
        # International transaction risk
        if 'international' in location or 'foreign' in location:
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    async def _get_ai_fraud_assessment(self, transaction_data: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Get AI-powered fraud risk assessment"""
        try:
            prompt = f"""
            Analyze this transaction for fraud risk:
            
            Amount: ₹{features['amount']:,}
            Time: {features['hour']:02d}:00 on {'weekend' if features['is_weekend'] else 'weekday'}
            Merchant: {features['merchant']}
            Location: {features['location']}
            Payment Method: {features.get('payment_method', 'card')}
            Account Age: {features.get('account_age', 0)} days
            Recent Transactions (24h): {features.get('transaction_count_24h', 0)}
            
            Consider:
            1. Transaction patterns and timing
            2. Amount relative to typical spending
            3. Merchant and location risk factors
            4. Account behavior patterns
            
            Rate fraud risk from 0.0 (no risk) to 1.0 (definite fraud).
            Focus on identifying specific suspicious indicators.
            """
            
            ai_response = await self.ai_service.generate_response(prompt, {
                'context': 'fraud_detection',
                'transaction_data': {k: v for k, v in features.items() if k not in ['user_history']}
            })
            
            response_text = ai_response.get('response', '').lower()
            
            # Parse AI risk assessment (simplified)
            if 'high risk' in response_text or 'definite fraud' in response_text:
                return 0.8
            elif 'moderate risk' in response_text or 'suspicious' in response_text:
                return 0.5
            elif 'low risk' in response_text:
                return 0.2
            else:
                return 0.3  # Default moderate risk
                
        except Exception as e:
            logger.error(f"Error getting AI fraud assessment: {e}")
            return 0.3  # Default risk score
    
    async def _get_anomaly_score(self, features: Dict[str, Any]) -> float:
        """Get anomaly score from machine learning model"""
        try:
            # Prepare features for anomaly detection
            feature_vector = [
                features['amount'],
                features['hour'],
                features['day_of_week'],
                0.5,  # merchant_risk placeholder
                0.3,  # location_risk placeholder
                features.get('account_age', 100),
                features.get('transaction_count_24h', 0) / 20.0  # normalized
            ]
            
            # Scale features
            feature_scaled = self.scaler.transform([feature_vector])
            
            # Get anomaly score
            anomaly_score = self.anomaly_model.decision_function(feature_scaled)[0]
            
            # Convert to probability (lower score = more anomalous)
            anomaly_prob = max(0.0, min(1.0, (0.5 - anomaly_score) / 1.0))
            
            return anomaly_prob
            
        except Exception as e:
            logger.error(f"Error calculating anomaly score: {e}")
            return 0.3
    
    async def _generate_fraud_factors(
        self, 
        transaction_data: Dict[str, Any], 
        features: Dict[str, Any], 
        risk_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate detailed fraud factors and explanations"""
        
        factors = {
            'fraud_probability': sum(risk_scores.values()) / len(risk_scores),
            'risk_level': 'high' if sum(risk_scores.values()) / len(risk_scores) > 0.7 else 
                         'medium' if sum(risk_scores.values()) / len(risk_scores) > 0.4 else 'low',
            'contributing_factors': [],
            'risk_breakdown': risk_scores,
            'recommendations': [],
            'confidence': min(0.95, max(0.6, sum(risk_scores.values()) / len(risk_scores)))
        }
        
        # Analyze contributing factors
        if risk_scores['amount_risk'] > 0.5:
            factors['contributing_factors'].append('Unusual transaction amount')
        
        if risk_scores['velocity_risk'] > 0.5:
            factors['contributing_factors'].append('High transaction velocity')
        
        if risk_scores['behavioral_risk'] > 0.5:
            factors['contributing_factors'].append('Suspicious account behavior')
        
        if risk_scores['merchant_risk'] > 0.5:
            factors['contributing_factors'].append('High-risk merchant/location')
        
        if risk_scores['ai_risk'] > 0.6:
            factors['contributing_factors'].append('AI detected suspicious patterns')
        
        # Generate recommendations
        if factors['fraud_probability'] > 0.8:
            factors['recommendations'] = [
                'Block transaction immediately',
                'Freeze account pending investigation',
                'Contact customer for verification',
                'Review recent account activity'
            ]
        elif factors['fraud_probability'] > 0.5:
            factors['recommendations'] = [
                'Require additional authentication',
                'Manual review recommended',
                'Monitor account closely',
                'Contact customer if amount is high'
            ]
        else:
            factors['recommendations'] = [
                'Transaction appears normal',
                'Continue monitoring',
                'No immediate action required'
            ]
        
        return factors
    
    def _is_within_timeframe(self, timestamp_str: str, hours: int = 24) -> bool:
        """Check if timestamp is within specified timeframe"""
        try:
            if not timestamp_str:
                return False
            tx_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return (datetime.now() - tx_time).total_seconds() < (hours * 3600)
        except:
            return False
    
    async def _fallback_fraud_detection(self, transaction_data: Dict[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """Fallback fraud detection when enhanced system fails"""
        amount = transaction_data.get('amount', 0)
        hour = transaction_data.get('time_of_day', 12)
        merchant = transaction_data.get('merchant', '').lower()
        
        risk_score = 0.0
        
        # Simple rules
        if amount > 100000:
            risk_score += 0.4
        if hour < 6 or hour > 22:
            risk_score += 0.3
        if any(word in merchant for word in ['casino', 'gambling', 'crypto']):
            risk_score += 0.4
        
        is_fraud = risk_score > 0.7
        
        factors = {
            'fraud_probability': risk_score,
            'risk_level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low',
            'contributing_factors': ['Rule-based assessment'],
            'recommendations': ['Manual review recommended'] if is_fraud else ['Monitor transaction'],
            'confidence': 0.6
        }
        
        return risk_score, is_fraud, factors
    
    async def analyze_user_fraud_patterns(self, user_id: str, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user's transaction patterns for fraud indicators"""
        if not transactions:
            return {'risk_level': 'unknown', 'patterns': []}
        
        patterns = []
        risk_indicators = 0
        
        # Analyze transaction timing
        hours = [tx.get('hour', 12) for tx in transactions if 'hour' in tx]
        if hours:
            night_transactions = sum(1 for h in hours if h < 6 or h > 22)
            if night_transactions > len(hours) * 0.3:
                patterns.append("Frequent late-night transactions")
                risk_indicators += 1
        
        # Analyze amounts
        amounts = [tx.get('amount', 0) for tx in transactions]
        if amounts:
            avg_amount = np.mean(amounts)
            recent_amounts = amounts[-5:]  # Last 5 transactions
            
            if any(amt > avg_amount * 3 for amt in recent_amounts):
                patterns.append("Recent transactions much higher than average")
                risk_indicators += 1
        
        # Analyze merchant patterns
        merchants = [tx.get('merchant', '').lower() for tx in transactions]
        high_risk_merchants = sum(1 for m in merchants 
                                 if any(word in m for word in ['casino', 'gambling', 'crypto']))
        
        if high_risk_merchants > 0:
            patterns.append(f"Transactions with {high_risk_merchants} high-risk merchants")
            risk_indicators += 1
        
        # Overall risk assessment
        if risk_indicators >= 3:
            risk_level = 'high'
        elif risk_indicators >= 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'user_id': user_id,
            'risk_level': risk_level,
            'risk_indicators': risk_indicators,
            'patterns': patterns,
            'total_transactions_analyzed': len(transactions),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def health_check(self) -> str:
        """Health check for fraud detection service"""
        try:
            # Test fraud detection
            test_data = {
                'amount': 5000,
                'time_of_day': 14,
                'location': 'normal_location',
                'merchant': 'regular_store',
                'user_history': [],
                'account_age': 200
            }
            
            fraud_prob, is_fraud, factors = await self.enhanced_fraud_detection(test_data)
            
            if 0 <= fraud_prob <= 1 and isinstance(is_fraud, bool) and isinstance(factors, dict):
                return "healthy"
            else:
                return "unhealthy"
        except Exception as e:
            logger.error(f"Fraud detection health check failed: {e}")
            return "unhealthy"
