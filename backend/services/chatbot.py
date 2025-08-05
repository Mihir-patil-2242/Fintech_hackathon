import os
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List, Tuple
from datetime import datetime
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedChatbotService:
    def __init__(self):
        # Load sentence transformer for better intent recognition
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded SentenceTransformer model successfully")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer: {e}")
            self.sentence_model = None
        
        # Enhanced intent patterns with embeddings
        self.intents = {
            'loan_status': {
                'patterns': [
                    "check my loan status",
                    "what's the status of my loan application",
                    "has my loan been approved",
                    "loan application status",
                    "where is my loan",
                    "is my loan approved"
                ],
                'responses': [
                    "I can help you check your loan status. Let me look up your recent applications.",
                    "Your loan application is currently being processed by our AI system. The decision should be available within minutes.",
                    "Based on your profile, your loan application shows: {status}. Would you like more details?"
                ]
            },
            'credit_score': {
                'patterns': [
                    "what's my credit score",
                    "check my credit rating",
                    "credit score information",
                    "how good is my credit",
                    "my creditworthiness",
                    "credit report"
                ],
                'responses': [
                    "Your current credit score is {credit_score}. This is calculated based on your payment history, credit utilization, and other factors.",
                    "Based on our AI analysis, your credit score is {credit_score}. This places you in the '{credit_rating}' category.",
                    "Your credit score has been updated to {credit_score}. Here are some tips to improve it further..."
                ]
            },
            'investment_advice': {
                'patterns': [
                    "investment advice",
                    "stock recommendations",
                    "should I invest in",
                    "market analysis",
                    "portfolio advice",
                    "trading suggestions",
                    "buy or sell stock"
                ],
                'responses': [
                    "Based on current market analysis, here are my investment recommendations: {recommendations}",
                    "Our AI has analyzed market trends and suggests: {investment_advice}",
                    "For your risk profile, I recommend: {personalized_advice}"
                ]
            },
            'fraud_report': {
                'patterns': [
                    "report fraud",
                    "suspicious transaction",
                    "unauthorized payment",
                    "someone used my account",
                    "fraudulent activity",
                    "security issue"
                ],
                'responses': [
                    "I take fraud reports very seriously. I've immediately flagged your account for security review.",
                    "Your fraud report has been logged with ID: {report_id}. Our security team will investigate within 24 hours.",
                    "I've initiated fraud protection protocols. Please check your recent transactions and confirm any unauthorized activity."
                ]
            },
            'account_balance': {
                'patterns': [
                    "account balance",
                    "check balance",
                    "how much money",
                    "current balance",
                    "account summary",
                    "available funds"
                ],
                'responses': [
                    "Your current account balance is ₹{balance:,}. This includes all pending transactions.",
                    "Account Summary: Available Balance: ₹{balance:,}, Pending: ₹{pending:,}",
                    "Your account shows ₹{balance:,} available. Would you like to see recent transactions?"
                ]
            },
            'transaction_history': {
                'patterns': [
                    "transaction history",
                    "recent transactions",
                    "payment history",
                    "show transactions",
                    "account activity",
                    "spending history"
                ],
                'responses': [
                    "Here are your recent transactions: {transactions}",
                    "Your transaction history shows {transaction_count} transactions in the last 30 days.",
                    "Recent activity: {recent_activity}. All transactions are secured on our blockchain."
                ]
            },
            'help': {
                'patterns': [
                    "help",
                    "what can you do",
                    "commands",
                    "assistance",
                    "how to use",
                    "features"
                ],
                'responses': [
                    "I can help with: ✅ Loan applications ✅ Credit scores ✅ Investment advice ✅ Fraud reporting ✅ Account inquiries ✅ Transaction history",
                    "Available services: Loan status checks, Credit analysis, Market insights, Security reports, Account management, and Transaction reviews.",
                    "I'm your AI financial assistant! Ask me about loans, investments, account balance, or report any concerns."
                ]
            },
            'greeting': {
                'patterns': [
                    "hello",
                    "hi",
                    "hey there",
                    "good morning",
                    "good afternoon",
                    "good evening",
                    "greetings"
                ],
                'responses': [
                    "Hello! I'm your AI financial assistant. How can I help you today?",
                    "Hi there! I'm here to help with all your banking and investment needs. What would you like to know?",
                    "Good day! I can assist with loans, investments, account queries, and more. How may I help?"
                ]
            },
            'market_data': {
                'patterns': [
                    "market data",
                    "stock prices",
                    "market trends",
                    "current market",
                    "stock market",
                    "market analysis"
                ],
                'responses': [
                    "Current market data shows: {market_summary}. Our AI recommends: {ai_recommendations}",
                    "Market trends indicate: {trends}. Based on technical analysis: {technical_analysis}",
                    "Live market data: {live_data}. Risk assessment: {risk_level}"
                ]
            }
        }
        
        # Pre-compute embeddings for intent patterns
        self.intent_embeddings = {}
        if self.sentence_model:
            self._compute_intent_embeddings()
    
    def _compute_intent_embeddings(self):
        """Pre-compute embeddings for all intent patterns"""
        try:
            for intent, data in self.intents.items():
                embeddings = self.sentence_model.encode(data['patterns'])
                self.intent_embeddings[intent] = embeddings
            logger.info("Intent embeddings computed successfully")
        except Exception as e:
            logger.error(f"Error computing intent embeddings: {e}")
    
    def process_message(self, user_id: str, message: str) -> Tuple[str, float, str]:
        """
        Enhanced message processing with better NLP
        Returns: (response_text, confidence_score, detected_intent)
        """
        # Clean and normalize message
        clean_message = message.lower().strip()
        
        # Detect intent using embeddings if available
        if self.sentence_model and self.intent_embeddings:
            intent, confidence = self._detect_intent_with_embeddings(clean_message)
        else:
            # Fallback to regex-based detection
            intent, confidence = self._detect_intent_with_regex(clean_message)
        
        # Generate contextual response
        response = self._generate_contextual_response(intent, clean_message, user_id)
        
        # Log conversation for analytics
        self._log_conversation(user_id, message, response, intent, confidence)
        
        return response, confidence, intent
    
    def _detect_intent_with_embeddings(self, message: str) -> Tuple[str, float]:
        """Detect intent using sentence embeddings and cosine similarity"""
        try:
            message_embedding = self.sentence_model.encode([message])
            
            best_intent = 'default'
            best_confidence = 0.0
            
            for intent, embeddings in self.intent_embeddings.items():
                # Calculate cosine similarity with all patterns
                similarities = cosine_similarity(message_embedding, embeddings)[0]
                max_similarity = np.max(similarities)
                
                if max_similarity > best_confidence:
                    best_intent = intent
                    best_confidence = max_similarity
            
            # Threshold for confidence
            if best_confidence < 0.3:
                best_intent = 'default'
                best_confidence = 0.3
            
            return best_intent, float(best_confidence)
            
        except Exception as e:
            logger.error(f"Error in embedding-based intent detection: {e}")
            return self._detect_intent_with_regex(message)
    
    def _detect_intent_with_regex(self, message: str) -> Tuple[str, float]:
        """Fallback regex-based intent detection"""
        best_intent = 'default'
        best_confidence = 0.0
        
        for intent, data in self.intents.items():
            for pattern in data['patterns']:
                pattern_words = set(pattern.lower().split())
                message_words = set(message.split())
                
                # Calculate word overlap
                overlap = len(pattern_words.intersection(message_words))
                confidence = overlap / len(pattern_words) if pattern_words else 0
                
                if confidence > best_confidence:
                    best_intent = intent
                    best_confidence = confidence
        
        # Boost confidence if specific keywords found
        if any(word in message for word in ['loan', 'credit', 'score', 'fraud', 'balance', 'invest']):
            best_confidence = min(best_confidence * 1.5, 0.95)
        
        return best_intent, best_confidence
    
    def _generate_contextual_response(self, intent: str, message: str, user_id: str) -> str:
        """Generate contextual response with dynamic data"""
        import random
        
        if intent not in self.intents:
            intent = 'default'
        
        if intent == 'default':
            return "I understand you're asking about financial services. Could you please be more specific? I can help with loans, credit scores, investments, fraud reporting, and account services."
        
        # Get base response template
        responses = self.intents[intent]['responses']
        base_response = random.choice(responses)
        
        # Add dynamic data based on intent
        try:
            if intent == 'credit_score':
                mock_score = abs(hash(user_id)) % 300 + 550  # Score between 550-850
                credit_rating = self._get_credit_rating(mock_score)
                base_response = base_response.format(
                    credit_score=mock_score,
                    credit_rating=credit_rating
                )
            
            elif intent == 'account_balance':
                mock_balance = abs(hash(user_id)) % 500000 + 50000
                mock_pending = abs(hash(user_id + "pending")) % 10000
                base_response = base_response.format(
                    balance=mock_balance,
                    pending=mock_pending
                )
            
            elif intent == 'investment_advice':
                recommendations = self._get_investment_recommendations(message)
                base_response = base_response.format(
                    recommendations=recommendations,
                    investment_advice=recommendations,
                    personalized_advice=recommendations
                )
            
            elif intent == 'fraud_report':
                report_id = f"FR{abs(hash(user_id + str(datetime.now())))%10000:04d}"
                base_response = base_response.format(report_id=report_id)
            
            elif intent == 'transaction_history':
                transaction_count = abs(hash(user_id)) % 50 + 10
                base_response = base_response.format(
                    transaction_count=transaction_count,
                    transactions="Recent payments to merchants, online transfers, and bill payments",
                    recent_activity="3 payments, 1 transfer, 2 deposits"
                )
            
            elif intent == 'market_data':
                base_response = base_response.format(
                    market_summary="Mixed trends with tech stocks up 2.3%",
                    ai_recommendations="Consider diversified portfolio",
                    trends="Bullish sentiment in fintech sector",
                    technical_analysis="RSI indicates oversold conditions",
                    live_data="NIFTY: 19,450 (+0.8%)",
                    risk_level="Moderate"
                )
                
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            # Return unformatted response if formatting fails
            pass
        
        return base_response
    
    def _get_credit_rating(self, score: int) -> str:
        """Convert credit score to rating"""
        if score >= 750:
            return "Excellent"
        elif score >= 700:
            return "Good"
        elif score >= 650:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _get_investment_recommendations(self, message: str) -> str:
        """Generate investment recommendations based on message content"""
        recommendations = []
        
        if any(word in message for word in ['safe', 'secure', 'low risk']):
            recommendations = ["Government bonds", "Blue-chip stocks", "Fixed deposits"]
        elif any(word in message for word in ['growth', 'aggressive', 'high return']):
            recommendations = ["Growth stocks", "Cryptocurrency", "Small-cap funds"]
        elif any(word in message for word in ['tech', 'technology', 'IT']):
            recommendations = ["Technology ETFs", "AI/ML companies", "Cloud computing stocks"]
        else:
            recommendations = ["Diversified mutual funds", "Index funds", "Balanced portfolio"]
        
        return ", ".join(recommendations[:3])
    
    def _log_conversation(self, user_id: str, message: str, response: str, 
                         intent: str, confidence: float):
        """Enhanced conversation logging with analytics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'message': message,
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'session_id': f"chat_{user_id}_{datetime.now().strftime('%Y%m%d')}",
            'message_length': len(message),
            'response_length': len(response),
            'processing_method': 'embeddings' if self.sentence_model else 'regex'
        }
        
        # In a production system, this would go to a proper logging system
        logger.info(f"Chat processed: {intent} (confidence: {confidence:.2f})")
        
        # Store in memory for session context (in production, use Redis/database)
        if not hasattr(self, 'conversation_history'):
            self.conversation_history = {}
        
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append(log_entry)
        
        # Keep only last 10 messages per user
        if len(self.conversation_history[user_id]) > 10:
            self.conversation_history[user_id] = self.conversation_history[user_id][-10:]
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        if not hasattr(self, 'conversation_history'):
            return []
        return self.conversation_history.get(user_id, [])
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary of all conversations"""
        if not hasattr(self, 'conversation_history'):
            return {'total_conversations': 0}
        
        total_conversations = sum(len(conv) for conv in self.conversation_history.values())
        unique_users = len(self.conversation_history)
        
        # Intent distribution
        intent_counts = {}
        confidence_scores = []
        
        for user_convs in self.conversation_history.values():
            for conv in user_convs:
                intent = conv['intent']
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
                confidence_scores.append(conv['confidence'])
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return {
            'total_conversations': total_conversations,
            'unique_users': unique_users,
            'intent_distribution': intent_counts,
            'average_confidence': float(avg_confidence),
            'processing_method': 'embeddings' if self.sentence_model else 'regex'
        }
    
    def process_follow_up(self, user_id: str, message: str) -> str:
        """Handle context-aware follow-up questions"""
        # Get recent conversation history
        history = self.get_conversation_history(user_id)
        
        if not history:
            # No context, process as new message
            response, _, _ = self.process_message(user_id, message)
            return response
        
        # Get last intent for context
        last_entry = history[-1]
        last_intent = last_entry['intent']
        
        # Context-aware responses
        if last_intent == 'investment_advice' and any(word in message.lower() for word in ['more', 'details', 'why', 'how']):
            return "Based on technical analysis: Current RSI levels suggest oversold conditions in tech sector. Market sentiment is bullish with strong institutional buying. Consider dollar-cost averaging for long-term investments."
        
        elif last_intent == 'credit_score' and any(word in message.lower() for word in ['improve', 'increase', 'better']):
            return "To improve your credit score: 1) Pay bills on time (35% impact) 2) Keep credit utilization below 30% 3) Don't close old credit accounts 4) Monitor your credit report regularly 5) Consider a secured credit card if needed."
        
        elif last_intent == 'loan_status' and any(word in message.lower() for word in ['when', 'how long', 'time']):
            return "Typical loan processing times: Pre-approval: 2-5 minutes (AI decision), Full approval: 1-3 business days, Disbursement: 1-7 days after approval. Your application is currently in the AI evaluation phase."
        
        # If no specific context match, process as new message
        response, _, _ = self.process_message(user_id, message)
        return response
