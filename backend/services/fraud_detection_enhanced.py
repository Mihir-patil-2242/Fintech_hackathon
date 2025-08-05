import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Tuple, List
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFraudDetectionService:
    def __init__(self):
        self.models_dir = "models"
        self.ensure_models_dir()
        
        # Models for different types of fraud detection
        self.anomaly_models = {}  # Unsupervised models
        self.supervised_models = {}  # Supervised models
        self.ensemble_model = None
        self.feature_scaler = StandardScaler()
        
        # Feature importance and model metrics
        self.feature_importance = {}
        self.model_metrics = {}
        
        # Fraud patterns and rules
        self.fraud_patterns = {}
        
        # Initialize models
        self._initialize_models()
    
    def ensure_models_dir(self):
        """Ensure models directory exists"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def _initialize_models(self):
        """Initialize and train fraud detection models"""
        try:
            self._load_pretrained_models()
            logger.info("Loaded pre-trained fraud detection models")
        except:
            logger.info("Pre-trained fraud models not found, training new models...")
            self._train_models()
            self._save_models()
    
    def _generate_fraud_training_data(self, n_samples: int = 20000) -> pd.DataFrame:
        """Generate synthetic training data for fraud detection"""
        np.random.seed(42)
        
        # Normal transactions (80% of data)
        n_normal = int(n_samples * 0.8)
        normal_data = {
            'amount': np.random.lognormal(6, 1.5, n_normal),  # Typical transaction amounts
            'hour': np.random.choice(range(6, 23), n_normal, p=[0.08]*17),  # Business hours
            'day_of_week': np.random.choice(range(7), n_normal),
            'merchant_risk_score': np.random.beta(2, 5, n_normal),  # Lower risk merchants
            'location_risk_score': np.random.beta(2, 8, n_normal),  # Lower risk locations
            'user_age_days': np.random.exponential(365, n_normal),  # Account age
            'avg_transaction_amount': np.random.lognormal(5.5, 1, n_normal),
            'transaction_frequency_24h': np.random.poisson(2, n_normal),
            'time_since_last_transaction': np.random.exponential(3600, n_normal),  # Seconds
            'payment_method_risk': np.random.beta(2, 5, n_normal),
            'velocity_score': np.random.beta(2, 8, n_normal),  # Transaction velocity
            'device_fingerprint_risk': np.random.beta(2, 10, n_normal),
            'geolocation_risk': np.random.beta(2, 8, n_normal),
            'is_fraud': np.zeros(n_normal)
        }
        
        # Fraudulent transactions (20% of data)
        n_fraud = n_samples - n_normal
        fraud_data = {
            'amount': np.concatenate([
                np.random.lognormal(8, 1, int(n_fraud * 0.7)),  # Large amounts
                np.random.uniform(1, 100, int(n_fraud * 0.3))   # Small test amounts
            ]),
            'hour': np.random.choice(range(24), n_fraud, 
                                   p=[0.08]*3 + [0.03]*3 + [0.04]*17 + [0.08]),  # Odd hours
            'day_of_week': np.random.choice(range(7), n_fraud),
            'merchant_risk_score': np.random.beta(5, 2, n_fraud),  # Higher risk merchants
            'location_risk_score': np.random.beta(5, 2, n_fraud),  # Higher risk locations
            'user_age_days': np.random.exponential(180, n_fraud),  # Newer accounts
            'avg_transaction_amount': np.random.lognormal(5, 1.5, n_fraud),
            'transaction_frequency_24h': np.random.poisson(8, n_fraud),  # High frequency
            'time_since_last_transaction': np.random.exponential(300, n_fraud),  # Quick succession
            'payment_method_risk': np.random.beta(5, 2, n_fraud),
            'velocity_score': np.random.beta(5, 2, n_fraud),  # High velocity
            'device_fingerprint_risk': np.random.beta(5, 2, n_fraud),
            'geolocation_risk': np.random.beta(5, 2, n_fraud),
            'is_fraud': np.ones(n_fraud)
        }
        
        # Combine and shuffle
        all_data = {}
        for key in normal_data.keys():
            all_data[key] = np.concatenate([normal_data[key], fraud_data[key]])
        
        df = pd.DataFrame(all_data)
        
        # Add derived features
        df['amount_z_score'] = (df['amount'] - df['avg_transaction_amount']) / (df['avg_transaction_amount'] + 1)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        df['high_velocity'] = (df['velocity_score'] > 0.7).astype(int)
        df['new_account'] = (df['user_age_days'] < 30).astype(int)
        df['risk_composite'] = (df['merchant_risk_score'] + df['location_risk_score'] + 
                               df['payment_method_risk'] + df['device_fingerprint_risk']) / 4
        
        return df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    def _train_models(self):
        """Train multiple fraud detection models"""
        logger.info("Generating fraud detection training data...")
        df = self._generate_fraud_training_data(25000)
        
        # Prepare features
        feature_cols = [
            'amount', 'hour', 'day_of_week', 'merchant_risk_score', 'location_risk_score',
            'user_age_days', 'avg_transaction_amount', 'transaction_frequency_24h',
            'time_since_last_transaction', 'payment_method_risk', 'velocity_score',
            'device_fingerprint_risk', 'geolocation_risk', 'amount_z_score',
            'is_weekend', 'is_night', 'high_velocity', 'new_account', 'risk_composite'
        ]
        
        X = df[feature_cols]
        y = df['is_fraud']
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, 
                                                           random_state=42, stratify=y)
        
        # Train unsupervised anomaly detection models
        logger.info("Training anomaly detection models...")
        X_normal = X_train[y_train == 0]  # Only normal transactions for unsupervised learning
        
        self.anomaly_models = {
            'isolation_forest': IsolationForest(contamination=0.15, random_state=42),
            'one_class_svm': OneClassSVM(nu=0.15),
        }
        
        for name, model in self.anomaly_models.items():
            model.fit(X_normal)
            # Test on full test set
            anomaly_pred = model.predict(X_test)
            anomaly_pred = (anomaly_pred == -1).astype(int)  # Convert to 0/1
            precision = precision_score(y_test, anomaly_pred, zero_division=0)
            recall = recall_score(y_test, anomaly_pred, zero_division=0)
            f1 = f1_score(y_test, anomaly_pred, zero_division=0)
            
            self.model_metrics[f'anomaly_{name}_precision'] = precision
            self.model_metrics[f'anomaly_{name}_recall'] = recall
            self.model_metrics[f'anomaly_{name}_f1'] = f1
            logger.info(f"Anomaly {name} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Train supervised classification models
        logger.info("Training supervised fraud detection models...")
        self.supervised_models = {
            'xgboost': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                       scale_pos_weight=4, random_state=42),  # Handle class imbalance
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                                  class_weight='balanced', random_state=42),
            'neural_net': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, 
                                      random_state=42),
        }
        
        for name, model in self.supervised_models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else pred
            
            accuracy = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred, zero_division=0)
            recall = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            
            self.model_metrics[f'supervised_{name}_accuracy'] = accuracy
            self.model_metrics[f'supervised_{name}_precision'] = precision
            self.model_metrics[f'supervised_{name}_recall'] = recall
            self.model_metrics[f'supervised_{name}_f1'] = f1
            
            logger.info(f"Supervised {name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
                       f"Recall: {recall:.3f}, F1: {f1:.3f}")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(feature_cols, model.feature_importances_))
        
        # Train ensemble model (best performing individual model)
        best_model_name = max(self.supervised_models.keys(), 
                            key=lambda x: self.model_metrics[f'supervised_{x}_f1'])
        self.ensemble_model = self.supervised_models[best_model_name]
        logger.info(f"Best model for ensemble: {best_model_name}")
        
        logger.info("Fraud detection model training completed!")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            joblib.dump(self.anomaly_models, os.path.join(self.models_dir, 'fraud_anomaly_models.pkl'))
            joblib.dump(self.supervised_models, os.path.join(self.models_dir, 'fraud_supervised_models.pkl'))
            joblib.dump(self.ensemble_model, os.path.join(self.models_dir, 'fraud_ensemble_model.pkl'))
            joblib.dump(self.feature_scaler, os.path.join(self.models_dir, 'fraud_feature_scaler.pkl'))
            joblib.dump(self.model_metrics, os.path.join(self.models_dir, 'fraud_model_metrics.pkl'))
            joblib.dump(self.feature_importance, os.path.join(self.models_dir, 'fraud_feature_importance.pkl'))
            logger.info("Fraud detection models saved successfully")
        except Exception as e:
            logger.error(f"Error saving fraud models: {e}")
    
    def _load_pretrained_models(self):
        """Load pre-trained models from disk"""
        self.anomaly_models = joblib.load(os.path.join(self.models_dir, 'fraud_anomaly_models.pkl'))
        self.supervised_models = joblib.load(os.path.join(self.models_dir, 'fraud_supervised_models.pkl'))
        self.ensemble_model = joblib.load(os.path.join(self.models_dir, 'fraud_ensemble_model.pkl'))
        self.feature_scaler = joblib.load(os.path.join(self.models_dir, 'fraud_feature_scaler.pkl'))
        self.model_metrics = joblib.load(os.path.join(self.models_dir, 'fraud_model_metrics.pkl'))
        self.feature_importance = joblib.load(os.path.join(self.models_dir, 'fraud_feature_importance.pkl'))
    
    def _extract_features(self, transaction_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from transaction data"""
        # Basic features
        amount = transaction_data.get('amount', 100)
        hour = transaction_data.get('time_of_day', 12)
        day_of_week = datetime.now().weekday()
        
        # Risk scores (would come from other services in production)
        merchant = transaction_data.get('merchant', '').lower()
        location = transaction_data.get('location', '').lower()
        
        merchant_risk = self._calculate_merchant_risk(merchant)
        location_risk = self._calculate_location_risk(location)
        
        # User history features
        user_history = transaction_data.get('user_history', [])
        user_age_days = transaction_data.get('account_age', 100)
        
        avg_transaction_amount = np.mean([tx.get('amount', 100) for tx in user_history]) if user_history else amount
        transaction_frequency_24h = len([tx for tx in user_history 
                                       if self._is_within_24h(tx.get('timestamp', ''))])
        
        last_tx_time = user_history[0].get('timestamp', '') if user_history else ''
        time_since_last = self._calculate_time_since_last_transaction(last_tx_time)
        
        # Additional risk factors
        payment_method_risk = self._calculate_payment_method_risk(
            transaction_data.get('payment_method', 'card')
        )
        velocity_score = min(transaction_frequency_24h / 10.0, 1.0)  # Normalize
        device_fingerprint_risk = np.random.beta(2, 8)  # Mock - would be real in production
        geolocation_risk = location_risk
        
        # Derived features
        amount_z_score = (amount - avg_transaction_amount) / (avg_transaction_amount + 1)
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night = 1 if hour < 6 or hour > 22 else 0
        high_velocity = 1 if velocity_score > 0.7 else 0
        new_account = 1 if user_age_days < 30 else 0
        risk_composite = (merchant_risk + location_risk + payment_method_risk + device_fingerprint_risk) / 4
        
        # Create feature array
        features = [
            amount, hour, day_of_week, merchant_risk, location_risk,
            user_age_days, avg_transaction_amount, transaction_frequency_24h,
            time_since_last, payment_method_risk, velocity_score,
            device_fingerprint_risk, geolocation_risk, amount_z_score,
            is_weekend, is_night, high_velocity, new_account, risk_composite
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_merchant_risk(self, merchant: str) -> float:
        """Calculate merchant risk score"""
        high_risk_keywords = ['casino', 'gambling', 'crypto', 'bitcoin', 'offshore', 'unknown']
        medium_risk_keywords = ['cash', 'advance', 'pawn', 'gold', 'atm']
        
        merchant_lower = merchant.lower()
        
        for keyword in high_risk_keywords:
            if keyword in merchant_lower:
                return np.random.uniform(0.7, 1.0)
        
        for keyword in medium_risk_keywords:
            if keyword in merchant_lower:
                return np.random.uniform(0.4, 0.7)
        
        return np.random.uniform(0.1, 0.4)
    
    def _calculate_location_risk(self, location: str) -> float:
        """Calculate location risk score"""
        high_risk_keywords = ['unknown', 'foreign', 'offshore', 'anonymous']
        medium_risk_keywords = ['airport', 'border', 'international']
        
        location_lower = location.lower()
        
        for keyword in high_risk_keywords:
            if keyword in location_lower:
                return np.random.uniform(0.7, 1.0)
        
        for keyword in medium_risk_keywords:
            if keyword in location_lower:
                return np.random.uniform(0.4, 0.7)
        
        return np.random.uniform(0.1, 0.4)
    
    def _calculate_payment_method_risk(self, payment_method: str) -> float:
        """Calculate payment method risk score"""
        risk_scores = {
            'crypto': 0.9,
            'prepaid': 0.7,
            'cash': 0.6,
            'wire': 0.5,
            'debit': 0.3,
            'credit': 0.2,
            'card': 0.25
        }
        return risk_scores.get(payment_method.lower(), 0.4)
    
    def _is_within_24h(self, timestamp_str: str) -> bool:
        """Check if timestamp is within last 24 hours"""
        try:
            if not timestamp_str:
                return False
            tx_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return (datetime.now() - tx_time).total_seconds() < 86400
        except:
            return False
    
    def _calculate_time_since_last_transaction(self, last_tx_timestamp: str) -> float:
        """Calculate time since last transaction in seconds"""
        try:
            if not last_tx_timestamp:
                return 86400  # 24 hours default
            last_time = datetime.fromisoformat(last_tx_timestamp.replace('Z', '+00:00'))
            return (datetime.now() - last_time).total_seconds()
        except:
            return 86400
    
    async def enhanced_fraud_detection(self, transaction_data: Dict[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """Enhanced fraud detection using ensemble of models"""
        try:
            # Extract and scale features
            X = self._extract_features(transaction_data)
            X_scaled = self.feature_scaler.transform(X)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            # Anomaly detection models
            for name, model in self.anomaly_models.items():
                anomaly_pred = model.predict(X_scaled)[0]
                # Convert to probability (anomaly = -1, normal = 1)
                anomaly_prob = 0.8 if anomaly_pred == -1 else 0.2
                predictions[f'anomaly_{name}'] = anomaly_prob
                probabilities[f'anomaly_{name}'] = anomaly_prob
            
            # Supervised models
            for name, model in self.supervised_models.items():
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_scaled)[0][1]  # Probability of fraud
                else:
                    prob = float(model.predict(X_scaled)[0])
                
                predictions[f'supervised_{name}'] = prob
                probabilities[f'supervised_{name}'] = prob
            
            # Ensemble prediction (weighted average)
            weights = {
                'anomaly_isolation_forest': 0.15,
                'anomaly_one_class_svm': 0.10,
                'supervised_xgboost': 0.35,
                'supervised_random_forest': 0.25,
                'supervised_neural_net': 0.15
            }
            
            fraud_probability = sum(probabilities.get(name, 0) * weight 
                                  for name, weight in weights.items())
            fraud_probability = max(0.0, min(1.0, fraud_probability))
            
            # Determine if fraud
            is_fraud = fraud_probability > 0.5
            
            # Generate fraud factors and explanations
            fraud_factors = self._generate_fraud_factors(transaction_data, X[0], fraud_probability)
            
            return fraud_probability, is_fraud, fraud_factors
            
        except Exception as e:
            logger.error(f"Error in enhanced fraud detection: {e}")
            # Fallback to simple rule-based detection
            return self._fallback_fraud_detection(transaction_data)
    
    def _generate_fraud_factors(self, transaction_data: Dict[str, Any], 
                               features: np.ndarray, fraud_prob: float) -> Dict[str, Any]:
        """Generate detailed fraud factors and explanations"""
        factors = {
            'fraud_probability': fraud_prob,
            'risk_level': 'high' if fraud_prob > 0.7 else 'medium' if fraud_prob > 0.4 else 'low',
            'confidence': min(0.95, abs(fraud_prob - 0.5) * 2),  # Higher confidence when far from 0.5
            'contributing_factors': [],
            'risk_indicators': {},
            'recommended_action': 'block' if fraud_prob > 0.8 else 'review' if fraud_prob > 0.5 else 'approve'
        }
        
        # Analyze individual risk factors
        amount = transaction_data.get('amount', 0)
        hour = transaction_data.get('time_of_day', 12)
        user_history = transaction_data.get('user_history', [])
        
        if amount > 10000:
            factors['contributing_factors'].append('Large transaction amount')
            factors['risk_indicators']['amount_risk'] = 'high'
        
        if hour < 6 or hour > 22:
            factors['contributing_factors'].append('Unusual transaction time')
            factors['risk_indicators']['time_risk'] = 'high'
        
        if len(user_history) > 5 and all(
            (datetime.now() - datetime.fromisoformat(tx.get('timestamp', '').replace('Z', '+00:00'))).total_seconds() < 3600 
            for tx in user_history[:5]
        ):
            factors['contributing_factors'].append('High transaction frequency')
            factors['risk_indicators']['velocity_risk'] = 'high'
        
        merchant = transaction_data.get('merchant', '').lower()
        if any(keyword in merchant for keyword in ['casino', 'gambling', 'crypto']):
            factors['contributing_factors'].append('High-risk merchant category')
            factors['risk_indicators']['merchant_risk'] = 'high'
        
        # Add model-specific insights
        if hasattr(self, 'feature_importance') and self.feature_importance:
            top_features = []
            for model_name, importance_dict in self.feature_importance.items():
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                top_features.extend([f[0] for f in sorted_features[:3]])
            
            factors['top_risk_features'] = list(set(top_features))
        
        return factors
    
    def _fallback_fraud_detection(self, transaction_data: Dict[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """Fallback rule-based fraud detection"""
        amount = transaction_data.get('amount', 0)
        hour = transaction_data.get('time_of_day', 12)
        location = transaction_data.get('location', '').lower()
        merchant = transaction_data.get('merchant', '').lower()
        
        risk_score = 0.0
        
        # Amount-based risk
        if amount > 50000:
            risk_score += 0.4
        elif amount > 10000:
            risk_score += 0.2
        
        # Time-based risk
        if hour < 6 or hour > 22:
            risk_score += 0.3
        
        # Location-based risk
        if any(keyword in location for keyword in ['unknown', 'foreign']):
            risk_score += 0.2
        
        # Merchant-based risk
        if any(keyword in merchant for keyword in ['casino', 'gambling', 'crypto']):
            risk_score += 0.3
        
        risk_score = min(1.0, risk_score)
        is_fraud = risk_score > 0.7
        
        factors = {
            'fraud_probability': risk_score,
            'risk_level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low',
            'confidence': 0.6,
            'contributing_factors': ['Rule-based assessment'],
            'recommended_action': 'review' if is_fraud else 'approve'
        }
        
        return risk_score, is_fraud, factors
    
    async def analyze_transaction_patterns(self, user_id: str, 
                                         transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user's transaction patterns for suspicious activity"""
        if not transactions:
            return {'pattern_risk': 0.0, 'anomalies': []}
        
        # Analyze patterns
        amounts = [tx.get('amount', 0) for tx in transactions]
        times = [tx.get('timestamp', '') for tx in transactions]
        
        # Statistical analysis
        avg_amount = np.mean(amounts)
        std_amount = np.std(amounts)
        
        anomalies = []
        pattern_risk = 0.0
        
        # Check for unusual amounts
        for tx in transactions:
            amount = tx.get('amount', 0)
            if abs(amount - avg_amount) > 3 * std_amount:
                anomalies.append(f"Unusual amount: ${amount:,.2f}")
                pattern_risk += 0.1
        
        # Check for rapid transactions
        sorted_times = sorted([datetime.fromisoformat(t.replace('Z', '+00:00')) for t in times if t])
        rapid_transactions = 0
        for i in range(1, len(sorted_times)):
            if (sorted_times[i] - sorted_times[i-1]).total_seconds() < 300:  # 5 minutes
                rapid_transactions += 1
        
        if rapid_transactions > 3:
            anomalies.append(f"Multiple rapid transactions: {rapid_transactions}")
            pattern_risk += 0.2
        
        pattern_risk = min(1.0, pattern_risk)
        
        return {
            'pattern_risk': pattern_risk,
            'anomalies': anomalies,
            'avg_amount': avg_amount,
            'transaction_velocity': len(transactions) / max(1, len(set(t[:10] for t in times if t))),  # Transactions per day
            'risk_assessment': 'high' if pattern_risk > 0.6 else 'medium' if pattern_risk > 0.3 else 'low'
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
        except Exception:
            return "unhealthy"
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get fraud detection model performance metrics"""
        return self.model_metrics.copy()
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance for interpretability"""
        return self.feature_importance.copy()
