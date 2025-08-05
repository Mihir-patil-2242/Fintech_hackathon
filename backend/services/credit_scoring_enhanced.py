import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Tuple, List
from datetime import datetime
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCreditScoringService:
    def __init__(self):
        self.models_dir = "models"
        self.ensure_models_dir()
        
        # Models
        self.credit_score_model = None
        self.loan_approval_model = None
        self.risk_assessment_model = None
        self.scaler = StandardScaler()
        
        # Model performance tracking
        self.model_metrics = {}
        
        # Initialize models
        self._initialize_models()
    
    def ensure_models_dir(self):
        """Ensure models directory exists"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def _initialize_models(self):
        """Initialize and train ML models if not already trained"""
        try:
            self._load_pretrained_models()
            logger.info("Loaded pre-trained credit scoring models")
        except:
            logger.info("Pre-trained models not found, training new models...")
            self._train_models()
            self._save_models()
    
    def _generate_training_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic training data for credit scoring"""
        np.random.seed(42)
        
        # Generate base features
        data = {
            'income': np.random.lognormal(10.5, 0.8, n_samples),  # Realistic income distribution
            'loan_amount': np.random.lognormal(11.5, 0.7, n_samples),
            'employment_years': np.random.exponential(5, n_samples).clip(0, 40),
            'credit_history_score': np.random.choice(range(1, 11), n_samples, 
                                                   p=[0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.05, 0.05]),
            'existing_debts': np.random.lognormal(9.0, 1.2, n_samples) * 0.3,
            'age': np.random.normal(35, 12, n_samples).clip(18, 70),
            'monthly_expenses': np.random.lognormal(8.5, 0.6, n_samples),
            'assets_value': np.random.lognormal(12.0, 1.5, n_samples) * 0.4,
        }
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['debt_to_income_ratio'] = df['existing_debts'] / df['income']
        df['loan_to_income_ratio'] = df['loan_amount'] / df['income']
        df['expense_to_income_ratio'] = df['monthly_expenses'] / (df['income'] / 12)
        df['assets_to_debt_ratio'] = df['assets_value'] / (df['existing_debts'] + 1)
        
        # Generate realistic credit scores using complex rules
        df['credit_score'] = (
            300 + 
            (df['credit_history_score'] * 45) +
            (np.log(df['income']) * 20) +
            (df['employment_years'] * 3) +
            (-df['debt_to_income_ratio'] * 200) +
            (-df['loan_to_income_ratio'] * 100) +
            (df['age'] * 2) +
            np.random.normal(0, 30, n_samples)
        ).clip(300, 850).astype(int)
        
        # Generate loan approval based on multiple factors
        approval_prob = (
            0.1 +  # Base probability
            (df['credit_score'] - 300) / 550 * 0.6 +  # Credit score impact
            (1 - df['debt_to_income_ratio'].clip(0, 1)) * 0.2 +  # Debt ratio impact
            (df['employment_years'] / 20).clip(0, 1) * 0.1  # Employment stability impact
        )
        
        df['loan_approved'] = np.random.binomial(1, approval_prob.clip(0, 1), n_samples)
        
        # Generate risk level (0-1, where 1 is highest risk)
        df['risk_level'] = (
            df['debt_to_income_ratio'] * 0.4 +
            (1 - (df['credit_score'] - 300) / 550) * 0.3 +
            df['loan_to_income_ratio'] * 0.2 +
            (1 - df['employment_years'] / 20).clip(0, 1) * 0.1
        ).clip(0, 1)
        
        return df
    
    def _train_models(self):
        """Train multiple ML models for credit scoring"""
        logger.info("Generating training data...")
        df = self._generate_training_data(15000)
        
        # Prepare features
        feature_cols = [
            'income', 'loan_amount', 'employment_years', 'credit_history_score',
            'existing_debts', 'age', 'monthly_expenses', 'assets_value',
            'debt_to_income_ratio', 'loan_to_income_ratio', 'expense_to_income_ratio',
            'assets_to_debt_ratio'
        ]
        
        X = df[feature_cols]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
        
        # Train credit score prediction model (regression)
        logger.info("Training credit score prediction model...")
        y_credit = df['credit_score']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_credit, test_size=0.2, random_state=42)
        
        # Ensemble of models for credit score prediction
        self.credit_score_model = {
            'xgboost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
            'neural_net': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }
        
        credit_predictions = {}
        for name, model in self.credit_score_model.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            credit_predictions[name] = pred
            self.model_metrics[f'credit_score_{name}_mse'] = mse
            logger.info(f"Credit score {name} MSE: {mse:.2f}")
        
        # Train loan approval model (classification)
        logger.info("Training loan approval model...")
        y_approval = df['loan_approved']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_approval, test_size=0.2, random_state=42)
        
        self.loan_approval_model = {
            'xgboost': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
            'neural_net': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        }
        
        for name, model in self.loan_approval_model.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            self.model_metrics[f'loan_approval_{name}_accuracy'] = accuracy
            logger.info(f"Loan approval {name} accuracy: {accuracy:.3f}")
        
        # Train risk assessment model (regression)
        logger.info("Training risk assessment model...")
        y_risk = df['risk_level']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_risk, test_size=0.2, random_state=42)
        
        self.risk_assessment_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        self.risk_assessment_model.fit(X_train, y_train)
        risk_pred = self.risk_assessment_model.predict(X_test)
        risk_mse = mean_squared_error(y_test, risk_pred)
        self.model_metrics['risk_assessment_mse'] = risk_mse
        logger.info(f"Risk assessment MSE: {risk_mse:.4f}")
        
        logger.info("Model training completed!")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            joblib.dump(self.credit_score_model, os.path.join(self.models_dir, 'credit_score_models.pkl'))
            joblib.dump(self.loan_approval_model, os.path.join(self.models_dir, 'loan_approval_models.pkl'))
            joblib.dump(self.risk_assessment_model, os.path.join(self.models_dir, 'risk_assessment_model.pkl'))
            joblib.dump(self.scaler, os.path.join(self.models_dir, 'feature_scaler.pkl'))
            joblib.dump(self.model_metrics, os.path.join(self.models_dir, 'model_metrics.pkl'))
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_pretrained_models(self):
        """Load pre-trained models from disk"""
        self.credit_score_model = joblib.load(os.path.join(self.models_dir, 'credit_score_models.pkl'))
        self.loan_approval_model = joblib.load(os.path.join(self.models_dir, 'loan_approval_models.pkl'))
        self.risk_assessment_model = joblib.load(os.path.join(self.models_dir, 'risk_assessment_model.pkl'))
        self.scaler = joblib.load(os.path.join(self.models_dir, 'feature_scaler.pkl'))
        self.model_metrics = joblib.load(os.path.join(self.models_dir, 'model_metrics.pkl'))
    
    def _prepare_features(self, loan_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for model prediction"""
        # Extract and compute features
        features = {
            'income': loan_data.get('income', 50000),
            'loan_amount': loan_data.get('loan_amount', 100000),
            'employment_years': loan_data.get('employment_years', 2),
            'credit_history_score': loan_data.get('credit_history_score', 5),
            'existing_debts': loan_data.get('existing_debts', 10000),
            'age': loan_data.get('age', 30),
            'monthly_expenses': loan_data.get('monthly_expenses', 3000),
            'assets_value': loan_data.get('assets_value', 50000),
        }
        
        # Derived features
        features['debt_to_income_ratio'] = features['existing_debts'] / features['income']
        features['loan_to_income_ratio'] = features['loan_amount'] / features['income']
        features['expense_to_income_ratio'] = features['monthly_expenses'] / (features['income'] / 12)
        features['assets_to_debt_ratio'] = features['assets_value'] / (features['existing_debts'] + 1)
        
        # Convert to array in correct order
        feature_order = [
            'income', 'loan_amount', 'employment_years', 'credit_history_score',
            'existing_debts', 'age', 'monthly_expenses', 'assets_value',
            'debt_to_income_ratio', 'loan_to_income_ratio', 'expense_to_income_ratio',
            'assets_to_debt_ratio'
        ]
        
        feature_array = np.array([features[col] for col in feature_order]).reshape(1, -1)
        
        # Scale features
        return self.scaler.transform(feature_array)
    
    async def calculate_enhanced_credit_score(self, loan_data: Dict[str, Any]) -> Tuple[int, float, List[str]]:
        """Calculate credit score using ensemble of ML models"""
        try:
            X = self._prepare_features(loan_data)
            
            # Get predictions from all models
            predictions = {}
            for name, model in self.credit_score_model.items():
                predictions[name] = model.predict(X)[0]
            
            # Ensemble prediction (weighted average)
            weights = {'xgboost': 0.4, 'neural_net': 0.3, 'gradient_boost': 0.3}
            credit_score = sum(predictions[name] * weights[name] for name in predictions)
            credit_score = int(np.clip(credit_score, 300, 850))
            
            # Calculate confidence based on model agreement
            std_dev = np.std(list(predictions.values()))
            confidence = max(0.5, 1.0 - (std_dev / 100))  # Higher agreement = higher confidence
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(loan_data, credit_score)
            
            return credit_score, confidence, risk_factors
            
        except Exception as e:
            logger.error(f"Error calculating credit score: {e}")
            # Fallback to simple calculation
            return self._fallback_credit_score(loan_data)
    
    async def enhanced_loan_approval(self, credit_score: int, loan_amount: float, 
                                   loan_data: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Enhanced loan approval using multiple ML models"""
        try:
            X = self._prepare_features(loan_data)
            
            # Get approval predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.loan_approval_model.items():
                pred = model.predict(X)[0]
                predictions[name] = pred
                
                # Get probability if available
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[0]
                    probabilities[name] = prob[1] if len(prob) > 1 else pred
                else:
                    probabilities[name] = float(pred)
            
            # Ensemble decision (weighted voting)
            weights = {'xgboost': 0.4, 'neural_net': 0.3, 'random_forest': 0.3}
            approval_prob = sum(probabilities[name] * weights[name] for name in probabilities)
            
            # Decision threshold
            approved = approval_prob > 0.5
            
            # Generate reason
            if approved:
                reason_factors = []
                if credit_score >= 700:
                    reason_factors.append("excellent credit score")
                elif credit_score >= 650:
                    reason_factors.append("good credit score")
                
                debt_ratio = loan_data.get('existing_debts', 0) / loan_data.get('income', 1)
                if debt_ratio < 0.3:
                    reason_factors.append("low debt-to-income ratio")
                
                if loan_data.get('employment_years', 0) >= 2:
                    reason_factors.append("stable employment history")
                
                reason = f"Approved based on: {', '.join(reason_factors) if reason_factors else 'overall financial profile'}"
            else:
                risk_factors = self._identify_risk_factors(loan_data, credit_score)
                reason = f"Declined due to: {', '.join(risk_factors[:2]) if risk_factors else 'insufficient creditworthiness'}"
            
            return approved, reason, approval_prob
            
        except Exception as e:
            logger.error(f"Error in loan approval: {e}")
            # Fallback to simple rule-based approval
            return self._fallback_loan_approval(credit_score, loan_amount)
    
    def _identify_risk_factors(self, loan_data: Dict[str, Any], credit_score: int) -> List[str]:
        """Identify risk factors based on loan data"""
        risk_factors = []
        
        if credit_score < 600:
            risk_factors.append("low credit score")
        
        debt_ratio = loan_data.get('existing_debts', 0) / loan_data.get('income', 1)
        if debt_ratio > 0.4:
            risk_factors.append("high debt-to-income ratio")
        
        loan_ratio = loan_data.get('loan_amount', 0) / loan_data.get('income', 1)
        if loan_ratio > 5:
            risk_factors.append("high loan-to-income ratio")
        
        if loan_data.get('employment_years', 0) < 1:
            risk_factors.append("limited employment history")
        
        if loan_data.get('credit_history_score', 5) < 4:
            risk_factors.append("poor credit history")
        
        return risk_factors
    
    def _fallback_credit_score(self, loan_data: Dict[str, Any]) -> Tuple[int, float, List[str]]:
        """Fallback credit score calculation"""
        income = loan_data.get('income', 50000)
        credit_history = loan_data.get('credit_history_score', 5)
        employment_years = loan_data.get('employment_years', 2)
        existing_debts = loan_data.get('existing_debts', 10000)
        
        base_score = 500
        income_score = min((income / 50000) * 100, 150)
        history_score = credit_history * 30
        employment_score = min(employment_years * 5, 50)
        debt_penalty = (existing_debts / income) * 200
        
        credit_score = int(base_score + income_score + history_score + employment_score - debt_penalty)
        credit_score = max(300, min(850, credit_score))
        
        risk_factors = self._identify_risk_factors(loan_data, credit_score)
        
        return credit_score, 0.7, risk_factors
    
    def _fallback_loan_approval(self, credit_score: int, loan_amount: float) -> Tuple[bool, str, float]:
        """Fallback loan approval logic"""
        if credit_score >= 700:
            return True, "Excellent credit score", 0.9
        elif credit_score >= 650 and loan_amount <= 500000:
            return True, "Good credit score for moderate loan", 0.7
        elif credit_score >= 600 and loan_amount <= 200000:
            return True, "Fair credit score for small loan", 0.6
        else:
            return False, f"Credit score {credit_score} too low for requested amount", 0.3
    
    async def get_risk_assessment(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive risk assessment"""
        try:
            X = self._prepare_features(loan_data)
            risk_score = self.risk_assessment_model.predict(X)[0]
            
            risk_level = "low" if risk_score < 0.3 else "medium" if risk_score < 0.7 else "high"
            
            return {
                "risk_score": float(risk_score),
                "risk_level": risk_level,
                "risk_factors": self._identify_risk_factors(loan_data, 700),  # Use average score for risk factors
                "mitigation_suggestions": self._get_mitigation_suggestions(risk_score)
            }
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return {"risk_score": 0.5, "risk_level": "medium", "risk_factors": [], "mitigation_suggestions": []}
    
    def _get_mitigation_suggestions(self, risk_score: float) -> List[str]:
        """Get risk mitigation suggestions"""
        suggestions = []
        
        if risk_score > 0.7:
            suggestions.extend([
                "Consider requiring a co-signer",
                "Request additional collateral",
                "Offer financial counseling services",
                "Implement enhanced monitoring"
            ])
        elif risk_score > 0.5:
            suggestions.extend([
                "Consider higher interest rate",
                "Shorter loan term",
                "Regular check-ins with customer"
            ])
        else:
            suggestions.append("Standard terms applicable")
        
        return suggestions
    
    async def health_check(self) -> str:
        """Health check for credit scoring service"""
        try:
            # Test model prediction
            test_data = {
                'income': 60000,
                'loan_amount': 150000,
                'employment_years': 3,
                'credit_history_score': 7,
                'existing_debts': 20000,
                'age': 30,
                'monthly_expenses': 3000,
                'assets_value': 100000
            }
            
            credit_score, confidence, risk_factors = await self.calculate_enhanced_credit_score(test_data)
            
            if 300 <= credit_score <= 850 and 0 <= confidence <= 1:
                return "healthy"
            else:
                return "unhealthy"
        except Exception:
            return "unhealthy"
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return self.model_metrics.copy()
