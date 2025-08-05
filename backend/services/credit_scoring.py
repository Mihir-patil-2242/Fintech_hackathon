import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Any

class CreditScoringService:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self._train_mock_model()
    
    def _train_mock_model(self):
        """Train a simple credit scoring model with mock data"""
        # Generate mock training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: income, loan_amount, credit_history, employment_years, debt_ratio
        data = {
            'income': np.random.normal(50000, 20000, n_samples),
            'loan_amount': np.random.normal(200000, 100000, n_samples),
            'credit_history_score': np.random.randint(1, 11, n_samples),
            'employment_years': np.random.randint(0, 30, n_samples),
            'debt_ratio': np.random.uniform(0, 1, n_samples)
        }
        
        df = pd.DataFrame(data)
        df['debt_ratio'] = df['existing_debts'] if 'existing_debts' in df.columns else df['debt_ratio']
        
        # Simple rule-based credit score generation
        df['credit_score'] = (
            (df['income'] / 1000 * 0.3) +
            (df['credit_history_score'] * 50) +
            (df['employment_years'] * 10) +
            (100 - df['debt_ratio'] * 100) +
            np.random.normal(0, 20, n_samples)
        ).clip(300, 850).astype(int)
        
        # This is a hackathon - use simple rules instead of ML for speed
        self.features = ['income', 'loan_amount', 'credit_history_score', 
                        'employment_years', 'debt_ratio']
    
    def calculate_credit_score(self, loan_data: Dict[str, Any]) -> int:
        """Calculate credit score using simple business rules"""
        income = loan_data.get('income', 0)
        loan_amount = loan_data.get('loan_amount', 0)
        credit_history = loan_data.get('credit_history_score', 5)
        employment_years = loan_data.get('employment_years', 0)
        existing_debts = loan_data.get('existing_debts', 0)
        
        # Calculate debt-to-income ratio
        debt_ratio = (existing_debts + loan_amount) / income if income > 0 else 1.0
        
        # Base score calculation
        base_score = 500
        
        # Income factor (higher income = better score)
        income_score = min((income / 50000) * 100, 150)
        
        # Credit history factor
        history_score = credit_history * 30
        
        # Employment stability factor
        employment_score = min(employment_years * 5, 50)
        
        # Debt ratio penalty
        debt_penalty = debt_ratio * 200
        
        # Final score
        credit_score = int(base_score + income_score + history_score + 
                          employment_score - debt_penalty)
        
        # Clamp to realistic range
        return max(300, min(850, credit_score))
    
    def approve_loan(self, credit_score: int, loan_amount: float) -> tuple[bool, str]:
        """Simple loan approval logic"""
        if credit_score >= 700:
            return True, "Excellent credit score"
        elif credit_score >= 650 and loan_amount <= 500000:
            return True, "Good credit score for moderate loan"
        elif credit_score >= 600 and loan_amount <= 200000:
            return True, "Fair credit score for small loan"
        else:
            return False, f"Credit score {credit_score} too low for requested amount"
