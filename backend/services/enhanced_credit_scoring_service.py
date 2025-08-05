import os
import logging
import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from .gemini_ai_service import GeminiAIService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCreditScoringService:
    def __init__(self):
        # Initialize AI service for advanced analysis
        self.ai_service = GeminiAIService()
        
        # Credit scoring parameters
        self.base_score = 500
        self.max_score = 850
        self.min_score = 300
        
        # Weight factors for credit scoring
        self.score_weights = {
            'payment_history': 0.35,
            'credit_utilization': 0.30,
            'credit_history_length': 0.15,
            'credit_mix': 0.10,
            'new_credit': 0.10
        }
        
        # Risk assessment thresholds
        self.risk_thresholds = {
            'excellent': 750,
            'good': 700,
            'fair': 650,
            'poor': 600
        }
        
        # External credit APIs (mock endpoints for demo)
        self.credit_apis = {
            'experian': os.getenv('EXPERIAN_API_KEY'),
            'equifax': os.getenv('EQUIFAX_API_KEY'),
            'transunion': os.getenv('TRANSUNION_API_KEY')
        }
        
        logger.info("Enhanced credit scoring service initialized")
    
    async def calculate_enhanced_credit_score(self, loan_data: Dict[str, Any]) -> Tuple[int, float, List[str]]:
        """Calculate enhanced credit score using multiple factors and AI analysis"""
        try:
            # Extract financial data
            income = loan_data.get('income', 0)
            loan_amount = loan_data.get('loan_amount', 0)
            employment_years = loan_data.get('employment_years', 0)
            existing_debts = loan_data.get('existing_debts', 0)
            credit_history_score = loan_data.get('credit_history_score', 5)
            age = loan_data.get('age', 30)
            
            # Calculate traditional credit score factors
            payment_score = await self._calculate_payment_history_score(loan_data)
            utilization_score = await self._calculate_credit_utilization_score(loan_data)
            history_score = await self._calculate_credit_history_score(loan_data)
            mix_score = await self._calculate_credit_mix_score(loan_data)
            new_credit_score = await self._calculate_new_credit_score(loan_data)
            
            # Calculate weighted score
            weighted_score = (
                payment_score * self.score_weights['payment_history'] +
                utilization_score * self.score_weights['credit_utilization'] +
                history_score * self.score_weights['credit_history_length'] +
                mix_score * self.score_weights['credit_mix'] +
                new_credit_score * self.score_weights['new_credit']
            )
            
            # Apply additional factors
            income_factor = min(1.2, income / 500000) if income > 0 else 0.8  # Income boost
            employment_factor = min(1.1, 1 + (employment_years * 0.02))  # Employment stability
            debt_factor = max(0.7, 1 - (existing_debts / income)) if income > 0 else 0.8  # Debt burden
            
            # Calculate final score
            final_score = weighted_score * income_factor * employment_factor * debt_factor
            final_score = int(np.clip(final_score, self.min_score, self.max_score))
            
            # Calculate confidence based on data completeness
            confidence = await self._calculate_confidence(loan_data)
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(loan_data, final_score)
            
            # Get AI insights for additional analysis
            ai_insights = await self._get_ai_credit_insights(loan_data, final_score)
            if ai_insights:
                risk_factors.extend(ai_insights.get('additional_risks', []))
            
            logger.info(f"Credit score calculated: {final_score} (confidence: {confidence:.2f})")
            return final_score, confidence, risk_factors
            
        except Exception as e:
            logger.error(f"Error calculating credit score: {e}")
            return await self._fallback_credit_score(loan_data)
    
    async def _calculate_payment_history_score(self, loan_data: Dict[str, Any]) -> float:
        """Calculate payment history score (35% of total score)"""
        credit_history_score = loan_data.get('credit_history_score', 5)
        
        # Convert 1-10 scale to score contribution
        base_contribution = 300  # Base payment history score
        
        if credit_history_score >= 8:
            return base_contribution + 200  # Excellent payment history
        elif credit_history_score >= 6:
            return base_contribution + 150  # Good payment history
        elif credit_history_score >= 4:
            return base_contribution + 100  # Fair payment history
        else:
            return base_contribution + 50   # Poor payment history
    
    async def _calculate_credit_utilization_score(self, loan_data: Dict[str, Any]) -> float:
        """Calculate credit utilization score (30% of total score)"""
        income = loan_data.get('income', 0)
        existing_debts = loan_data.get('existing_debts', 0)
        loan_amount = loan_data.get('loan_amount', 0)
        
        if income <= 0:
            return 200  # Default score for missing income data
        
        # Calculate debt-to-income ratio
        total_debt = existing_debts + loan_amount
        utilization_ratio = total_debt / income
        
        base_contribution = 250
        
        if utilization_ratio <= 0.1:
            return base_contribution + 100  # Excellent utilization
        elif utilization_ratio <= 0.3:
            return base_contribution + 75   # Good utilization
        elif utilization_ratio <= 0.5:
            return base_contribution + 25   # Fair utilization
        else:
            return base_contribution - 50   # High utilization
    
    async def _calculate_credit_history_score(self, loan_data: Dict[str, Any]) -> float:
        """Calculate credit history length score (15% of total score)"""
        age = loan_data.get('age', 30)
        employment_years = loan_data.get('employment_years', 0)
        
        # Estimate credit history length
        estimated_credit_age = max(0, age - 18)  # Assume credit started at 18
        
        base_contribution = 100
        
        if estimated_credit_age >= 15:
            return base_contribution + 50   # Long credit history
        elif estimated_credit_age >= 7:
            return base_contribution + 35   # Moderate credit history
        elif estimated_credit_age >= 2:
            return base_contribution + 20   # Short credit history
        else:
            return base_contribution        # Very short credit history
    
    async def _calculate_credit_mix_score(self, loan_data: Dict[str, Any]) -> float:
        """Calculate credit mix score (10% of total score)"""
        # In a real system, this would analyze different types of credit accounts
        # For demo, we'll use employment type and loan purpose as proxies
        
        employment_type = loan_data.get('employment_type', 'salaried')
        purpose = loan_data.get('purpose', 'personal')
        
        base_contribution = 70
        
        # Assume diversified financial behavior based on employment and purpose
        if employment_type in ['business', 'self-employed'] and purpose in ['business', 'home']:
            return base_contribution + 30   # Good credit mix
        elif purpose in ['car', 'education', 'home']:
            return base_contribution + 20   # Moderate credit mix
        else:
            return base_contribution + 10   # Limited credit mix
    
    async def _calculate_new_credit_score(self, loan_data: Dict[str, Any]) -> float:
        """Calculate new credit inquiries score (10% of total score)"""
        # In a real system, this would check recent credit inquiries
        # For demo, we'll estimate based on current application timing
        
        base_contribution = 70
        
        # Assume this is not the first recent application
        return base_contribution + 20  # Moderate new credit activity
    
    async def _calculate_confidence(self, loan_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on data completeness"""
        required_fields = ['income', 'loan_amount', 'employment_years', 'existing_debts', 'credit_history_score']
        optional_fields = ['age', 'purpose', 'employment_type', 'assets_value']
        
        required_completeness = sum(1 for field in required_fields if loan_data.get(field) is not None) / len(required_fields)
        optional_completeness = sum(1 for field in optional_fields if loan_data.get(field) is not None) / len(optional_fields)
        
        # Weight required fields more heavily
        overall_completeness = (required_completeness * 0.8) + (optional_completeness * 0.2)
        
        # Base confidence starts at 0.6, increases with data completeness
        confidence = 0.6 + (overall_completeness * 0.35)
        
        return min(0.95, confidence)
    
    async def _identify_risk_factors(self, loan_data: Dict[str, Any], credit_score: int) -> List[str]:
        """Identify risk factors based on loan data and credit score"""
        risk_factors = []
        
        income = loan_data.get('income', 0)
        loan_amount = loan_data.get('loan_amount', 0)
        existing_debts = loan_data.get('existing_debts', 0)
        employment_years = loan_data.get('employment_years', 0)
        credit_history_score = loan_data.get('credit_history_score', 5)
        
        # Credit score risks
        if credit_score < 600:
            risk_factors.append("Low credit score indicates higher default risk")
        elif credit_score < 650:
            risk_factors.append("Below-average credit score")
        
        # Income vs debt risks
        if income > 0:
            debt_ratio = (existing_debts + loan_amount) / income
            if debt_ratio > 0.5:
                risk_factors.append("High debt-to-income ratio")
            elif debt_ratio > 0.4:
                risk_factors.append("Elevated debt-to-income ratio")
        
        # Employment stability risks
        if employment_years < 1:
            risk_factors.append("Limited employment history")
        elif employment_years < 2:
            risk_factors.append("Short employment tenure")
        
        # Credit history risks
        if credit_history_score < 4:
            risk_factors.append("Poor payment history")
        elif credit_history_score < 6:
            risk_factors.append("Mixed payment history")
        
        # Loan amount risks
        if income > 0 and loan_amount > income * 3:
            risk_factors.append("Loan amount high relative to income")
        
        return risk_factors
    
    async def _get_ai_credit_insights(self, loan_data: Dict[str, Any], credit_score: int) -> Optional[Dict[str, Any]]:
        """Get AI-powered insights for credit assessment"""
        try:
            prompt = f"""
            Analyze this credit profile for additional insights:
            
            Credit Score: {credit_score}
            Income: ₹{loan_data.get('income', 0):,}
            Loan Amount: ₹{loan_data.get('loan_amount', 0):,}
            Employment Years: {loan_data.get('employment_years', 0)}
            Existing Debts: ₹{loan_data.get('existing_debts', 0):,}
            Credit History Score: {loan_data.get('credit_history_score', 5)}/10
            Purpose: {loan_data.get('purpose', 'Not specified')}
            
            Provide insights on:
            1. Additional risk factors not captured in traditional scoring
            2. Recommendations for improving creditworthiness
            3. Market conditions impact on this profile
            
            Keep response concise and professional.
            """
            
            ai_response = await self.ai_service.generate_response(prompt, {
                'context': 'credit_analysis',
                'score': credit_score,
                'loan_data': loan_data
            })
            
            return {
                'ai_insights': ai_response.get('response', ''),
                'additional_risks': [],  # Would parse from AI response
                'recommendations': []    # Would parse from AI response
            }
            
        except Exception as e:
            logger.error(f"Error getting AI credit insights: {e}")
            return None
    
    async def enhanced_loan_approval(
        self, 
        credit_score: int, 
        loan_amount: float, 
        loan_data: Dict[str, Any]
    ) -> Tuple[bool, str, float]:
        """Enhanced loan approval decision with AI assistance"""
        try:
            # Basic approval logic
            base_approval, base_reason = self._basic_approval_logic(credit_score, loan_amount, loan_data)
            
            # Get AI recommendation
            ai_recommendation = await self._get_ai_loan_recommendation(credit_score, loan_amount, loan_data)
            
            # Combine decisions
            if ai_recommendation:
                ai_approved = ai_recommendation.get('approved', base_approval)
                ai_reason = ai_recommendation.get('reason', base_reason)
                ai_confidence = ai_recommendation.get('confidence', 0.7)
                
                # Final decision (can be more sophisticated)
                final_approved = base_approval and ai_approved
                final_reason = ai_reason if ai_confidence > 0.7 else base_reason
                final_confidence = (ai_confidence + 0.8) / 2  # Average with base confidence
            else:
                final_approved = base_approval
                final_reason = base_reason
                final_confidence = 0.75
            
            return final_approved, final_reason, final_confidence
            
        except Exception as e:
            logger.error(f"Error in loan approval: {e}")
            return self._fallback_loan_approval(credit_score, loan_amount)
    
    def _basic_approval_logic(self, credit_score: int, loan_amount: float, loan_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Basic rule-based loan approval logic"""
        income = loan_data.get('income', 0)
        existing_debts = loan_data.get('existing_debts', 0)
        employment_years = loan_data.get('employment_years', 0)
        
        # Income requirements
        if income < 300000:  # Minimum income requirement
            return False, "Income below minimum requirement (₹3,00,000)"
        
        # Credit score requirements
        if credit_score < 550:
            return False, "Credit score too low for loan approval"
        
        # Debt-to-income ratio
        if income > 0:
            total_debt_ratio = (existing_debts + loan_amount) / income
            if total_debt_ratio > 0.6:
                return False, "Debt-to-income ratio too high"
        
        # Employment stability
        if employment_years < 1:
            return False, "Insufficient employment history"
        
        # Loan amount vs income
        if loan_amount > income * 5:
            return False, "Loan amount too high relative to income"
        
        # Approval conditions
        if credit_score >= 750:
            return True, "Excellent credit profile - approved"
        elif credit_score >= 700 and loan_amount <= income * 3:
            return True, "Good credit score and reasonable loan amount"
        elif credit_score >= 650 and loan_amount <= income * 2:
            return True, "Approved with standard terms"
        elif credit_score >= 600 and loan_amount <= income * 1.5:
            return True, "Approved with higher interest rate"
        else:
            return False, "Credit profile does not meet approval criteria"
    
    async def _get_ai_loan_recommendation(
        self, 
        credit_score: int, 
        loan_amount: float, 
        loan_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get AI recommendation for loan approval"""
        try:
            prompt = f"""
            As a loan underwriter, evaluate this loan application:
            
            Credit Score: {credit_score}
            Loan Amount: ₹{loan_amount:,}
            Monthly Income: ₹{loan_data.get('income', 0)/12:,.0f}
            Employment Years: {loan_data.get('employment_years', 0)}
            Existing Debts: ₹{loan_data.get('existing_debts', 0):,}
            Purpose: {loan_data.get('purpose', 'Not specified')}
            
            Consider current market conditions, economic factors, and regulatory requirements.
            
            Provide:
            1. Approval recommendation (YES/NO)
            2. Key reasoning
            3. Risk assessment
            4. Suggested terms if approved
            
            Be conservative but fair in assessment.
            """
            
            ai_response = await self.ai_service.generate_response(prompt, {
                'context': 'loan_underwriting',
                'credit_score': credit_score,
                'loan_amount': loan_amount
            })
            
            response_text = ai_response.get('response', '').lower()
            
            # Parse AI response (simplified)
            approved = 'approve' in response_text or 'yes' in response_text[:100]
            confidence = ai_response.get('confidence', 0.7)
            
            return {
                'approved': approved,
                'reason': ai_response.get('response', 'AI analysis completed'),
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error getting AI loan recommendation: {e}")
            return None
    
    async def _fallback_credit_score(self, loan_data: Dict[str, Any]) -> Tuple[int, float, List[str]]:
        """Fallback credit score calculation"""
        income = loan_data.get('income', 50000)
        credit_history = loan_data.get('credit_history_score', 5)
        employment_years = loan_data.get('employment_years', 2)
        existing_debts = loan_data.get('existing_debts', 10000)
        
        base_score = 500
        income_score = min((income / 50000) * 100, 150)
        history_score = credit_history * 30
        employment_score = min(employment_years * 5, 50)
        debt_penalty = (existing_debts / income) * 200 if income > 0 else 50
        
        credit_score = int(base_score + income_score + history_score + employment_score - debt_penalty)
        credit_score = max(300, min(850, credit_score))
        
        risk_factors = await self._identify_risk_factors(loan_data, credit_score)
        
        return credit_score, 0.7, risk_factors
    
    def _fallback_loan_approval(self, credit_score: int, loan_amount: float) -> Tuple[bool, str, float]:
        """Fallback loan approval logic"""
        if credit_score >= 700:
            return True, "Approved based on credit score", 0.8
        elif credit_score >= 650 and loan_amount <= 500000:
            return True, "Approved with conditions", 0.7
        elif credit_score >= 600 and loan_amount <= 200000:
            return True, "Approved for reduced amount", 0.6
        else:
            return False, f"Credit score {credit_score} below approval threshold", 0.8
    
    async def get_credit_improvement_suggestions(self, loan_data: Dict[str, Any], credit_score: int) -> List[str]:
        """Get suggestions for improving credit score"""
        suggestions = []
        
        # Based on current score
        if credit_score < 650:
            suggestions.extend([
                "Pay all bills on time for the next 6 months",
                "Reduce credit card balances to below 30% of limits",
                "Consider a secured credit card to build history"
            ])
        
        # Based on specific factors
        income = loan_data.get('income', 0)
        existing_debts = loan_data.get('existing_debts', 0)
        
        if income > 0 and existing_debts / income > 0.4:
            suggestions.append("Focus on reducing existing debt burden")
        
        employment_years = loan_data.get('employment_years', 0)
        if employment_years < 2:
            suggestions.append("Maintain stable employment for better credit profile")
        
        # Get AI-powered suggestions
        try:
            ai_suggestions = await self._get_ai_improvement_suggestions(loan_data, credit_score)
            if ai_suggestions:
                suggestions.extend(ai_suggestions)
        except Exception as e:
            logger.error(f"Error getting AI suggestions: {e}")
        
        return suggestions[:6]  # Limit to 6 suggestions
    
    async def _get_ai_improvement_suggestions(self, loan_data: Dict[str, Any], credit_score: int) -> List[str]:
        """Get AI-powered credit improvement suggestions"""
        try:
            prompt = f"""
            Provide 3-4 specific, actionable suggestions to improve a credit score of {credit_score}.
            
            Current profile:
            - Income: ₹{loan_data.get('income', 0):,}
            - Existing Debts: ₹{loan_data.get('existing_debts', 0):,}
            - Employment Years: {loan_data.get('employment_years', 0)}
            
            Focus on practical steps that can be implemented within 6-12 months.
            """
            
            ai_response = await self.ai_service.generate_response(prompt)
            response_text = ai_response.get('response', '')
            
            # Parse suggestions (simplified - in production would use better parsing)
            suggestions = []
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                    clean_suggestion = line.lstrip('-•0123456789. ').strip()
                    if clean_suggestion:
                        suggestions.append(clean_suggestion)
            
            return suggestions[:4]
            
        except Exception as e:
            logger.error(f"Error getting AI improvement suggestions: {e}")
            return []
    
    async def health_check(self) -> str:
        """Health check for credit scoring service"""
        try:
            # Test credit score calculation
            test_data = {
                'income': 60000,
                'loan_amount': 150000,
                'employment_years': 3,
                'credit_history_score': 7,
                'existing_debts': 20000,
                'age': 30
            }
            
            credit_score, confidence, risk_factors = await self.calculate_enhanced_credit_score(test_data)
            
            if 300 <= credit_score <= 850 and 0 <= confidence <= 1:
                return "healthy"
            else:
                return "unhealthy"
        except Exception:
            return "unhealthy"
