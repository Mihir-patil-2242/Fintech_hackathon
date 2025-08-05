import os
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import asyncio
import uuid
from .gemini_ai_service import GeminiAIService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedChatbotService:
    def __init__(self):
        # Initialize AI service
        self.ai_service = GeminiAIService()
        
        # Conversation context storage (use Redis in production)
        self.conversation_context = {}
        self.conversation_history = {}
        
        # Intent classification patterns
        self.intent_patterns = {
            'loan_inquiry': [
                'loan', 'borrow', 'mortgage', 'credit', 'financing', 'apply',
                'eligibility', 'interest rate', 'emi', 'repayment'
            ],
            'investment_advice': [
                'invest', 'stock', 'mutual fund', 'portfolio', 'market', 'trading',
                'buy', 'sell', 'returns', 'dividend', 'equity', 'bonds'
            ],
            'account_inquiry': [
                'balance', 'statement', 'transaction', 'account', 'deposit',
                'withdrawal', 'bank', 'savings', 'current'
            ],
            'fraud_report': [
                'fraud', 'suspicious', 'unauthorized', 'scam', 'phishing',
                'security', 'hack', 'stolen', 'compromised'
            ],
            'credit_score': [
                'credit score', 'cibil', 'rating', 'creditworthiness',
                'credit report', 'credit history'
            ],
            'general_inquiry': [
                'help', 'information', 'what', 'how', 'when', 'where', 'why'
            ]
        }
        
        # Quick action suggestions based on intent
        self.intent_actions = {
            'loan_inquiry': [
                'Check loan eligibility',
                'Calculate EMI',
                'Apply for loan',
                'Compare loan options'
            ],
            'investment_advice': [
                'View portfolio',
                'Get market insights',
                'Stock recommendations',
                'Risk assessment'
            ],
            'account_inquiry': [
                'Check balance',
                'View transactions',
                'Download statement',
                'Update profile'
            ],
            'fraud_report': [
                'Report fraud',
                'Block card',
                'Change password',
                'Contact security'
            ],
            'credit_score': [
                'Check credit score',
                'Improve credit rating',
                'Credit report',
                'Dispute errors'
            ]
        }
        
        logger.info("Enhanced chatbot service initialized")
    
    async def process_enhanced_message(
        self, 
        user_id: str, 
        message: str, 
        session_id: Optional[str] = None
    ) -> Tuple[str, float, str, List[str]]:
        """
        Process message with enhanced AI and context awareness
        Returns: (response_text, confidence, intent, suggested_actions)
        """
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Detect intent
            intent = await self._detect_intent(message)
            
            # Get conversation context
            context = await self._get_conversation_context(user_id, session_id)
            
            # Enhance context with user profile and intent
            enhanced_context = await self._enhance_context(context, user_id, intent, message)
            
            # Generate AI response
            ai_response = await self.ai_service.generate_response(message, enhanced_context)
            
            response_text = ai_response.get('response', 'I apologize, but I cannot process your request right now.')
            confidence = ai_response.get('confidence', 0.5)
            
            # Get suggested actions
            suggested_actions = self._get_suggested_actions(intent, message)
            
            # Update conversation history
            await self._update_conversation_history(user_id, session_id, message, response_text, intent, confidence)
            
            # Log interaction
            await self._log_conversation(user_id, session_id, message, response_text, intent, confidence)
            
            return response_text, confidence, intent, suggested_actions
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            fallback_response = "I apologize for the technical difficulty. Please try rephrasing your question or contact our support team."
            return fallback_response, 0.3, 'error', ['Contact support', 'Try again']
    
    async def _detect_intent(self, message: str) -> str:
        """Detect user intent from message"""
        message_lower = message.lower()
        intent_scores = {}
        
        # Calculate scores for each intent
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in message_lower:
                    score += 1
            
            # Normalize score
            if patterns:
                intent_scores[intent] = score / len(patterns)
        
        # Get intent with highest score
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[best_intent] > 0:
                return best_intent
        
        return 'general_inquiry'
    
    async def _get_conversation_context(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get conversation context for the user session"""
        context_key = f"{user_id}_{session_id}"
        
        if context_key not in self.conversation_context:
            self.conversation_context[context_key] = {
                'session_id': session_id,
                'user_id': user_id,
                'start_time': datetime.now().isoformat(),
                'message_count': 0,
                'topics_discussed': [],
                'user_preferences': {}
            }
        
        return self.conversation_context[context_key]
    
    async def _enhance_context(self, context: Dict[str, Any], user_id: str, intent: str, message: str) -> Dict[str, Any]:
        """Enhance context with additional information"""
        enhanced_context = context.copy()
        
        # Add current message info
        enhanced_context.update({
            'current_intent': intent,
            'current_message': message,
            'timestamp': datetime.now().isoformat(),
            'message_count': context.get('message_count', 0) + 1
        })
        
        # Add intent-specific context
        if intent == 'loan_inquiry':
            enhanced_context['loan_context'] = {
                'user_type': 'existing_customer',  # Would fetch from user service
                'previous_loans': 'none',  # Would fetch from database
                'credit_score_range': '650-750'  # Would fetch from credit service
            }
        elif intent == 'investment_advice':
            enhanced_context['investment_context'] = {
                'risk_tolerance': 'moderate',  # Would fetch from user profile
                'investment_experience': 'beginner',
                'portfolio_value': 'not_disclosed'
            }
        elif intent == 'account_inquiry':
            enhanced_context['account_context'] = {
                'account_type': 'savings',  # Would fetch from user service
                'account_status': 'active',
                'recent_activity': 'normal'
            }
        
        # Add topics to discussion history
        if intent not in enhanced_context.get('topics_discussed', []):
            enhanced_context.setdefault('topics_discussed', []).append(intent)
        
        return enhanced_context
    
    def _get_suggested_actions(self, intent: str, message: str) -> List[str]:
        """Get suggested actions based on intent"""
        actions = self.intent_actions.get(intent, ['Get help', 'Contact support'])
        
        # Add context-specific actions
        message_lower = message.lower()
        
        if 'apply' in message_lower and intent == 'loan_inquiry':
            actions.insert(0, 'Start loan application')
        elif 'check' in message_lower and intent == 'account_inquiry':
            actions.insert(0, 'View account details')
        elif 'report' in message_lower and intent == 'fraud_report':
            actions.insert(0, 'File fraud report')
        
        return actions[:4]  # Limit to 4 suggestions
    
    async def _update_conversation_history(
        self,
        user_id: str,
        session_id: str,
        message: str,
        response: str,
        intent: str,
        confidence: float
    ):
        """Update conversation history"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'user_message': message,
            'bot_response': response,
            'intent': intent,
            'confidence': confidence,
            'message_id': str(uuid.uuid4())
        }
        
        self.conversation_history[user_id].append(conversation_entry)
        
        # Keep only last 50 messages per user
        if len(self.conversation_history[user_id]) > 50:
            self.conversation_history[user_id] = self.conversation_history[user_id][-50:]
        
        # Update context
        context_key = f"{user_id}_{session_id}"
        if context_key in self.conversation_context:
            self.conversation_context[context_key]['message_count'] += 1
            self.conversation_context[context_key]['last_activity'] = datetime.now().isoformat()
    
    async def _log_conversation(
        self,
        user_id: str,
        session_id: str,
        message: str,
        response: str,
        intent: str,
        confidence: float
    ):
        """Log conversation for analytics and audit"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'session_id': session_id,
            'message': message,
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'service': 'enhanced_chatbot',
            'ai_model': 'gemini-2.0-flash-exp'
        }
        
        # In production, this would go to a logging service or database
        logger.info(f"Chat processed: {intent} (confidence: {confidence:.2f}) for user {user_id}")
    
    async def get_user_chat_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's chat history"""
        if user_id not in self.conversation_history:
            return []
        
        history = self.conversation_history[user_id]
        return history[-limit:] if limit > 0 else history
    
    async def get_conversation_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get conversation analytics for a user"""
        if user_id not in self.conversation_history:
            return {
                'total_messages': 0,
                'sessions': 0,
                'intents': {},
                'avg_confidence': 0
            }
        
        history = self.conversation_history[user_id]
        
        # Calculate analytics
        sessions = set(entry['session_id'] for entry in history)
        intents = {}
        confidences = []
        
        for entry in history:
            intent = entry['intent']
            intents[intent] = intents.get(intent, 0) + 1
            confidences.append(entry['confidence'])
        
        return {
            'total_messages': len(history),
            'sessions': len(sessions),
            'intents': intents,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'most_common_intent': max(intents, key=intents.get) if intents else None
        }
    
    async def clear_user_context(self, user_id: str, session_id: Optional[str] = None):
        """Clear user conversation context"""
        if session_id:
            context_key = f"{user_id}_{session_id}"
            if context_key in self.conversation_context:
                del self.conversation_context[context_key]
        else:
            # Clear all contexts for user
            keys_to_remove = [key for key in self.conversation_context.keys() if key.startswith(f"{user_id}_")]
            for key in keys_to_remove:
                del self.conversation_context[key]
    
    async def health_check(self) -> str:
        """Health check for chatbot service"""
        try:
            # Test AI service
            ai_health = await self.ai_service.health_check()
            
            # Test intent detection
            test_intent = await self._detect_intent("I want to apply for a loan")
            
            # Test context management
            test_context = await self._get_conversation_context("test_user", "test_session")
            
            if ai_health == "healthy" and test_intent and test_context:
                return "healthy"
            else:
                return "degraded"
                
        except Exception as e:
            logger.error(f"Chatbot health check failed: {e}")
            return "unhealthy"
    
    def get_supported_intents(self) -> List[str]:
        """Get list of supported intents"""
        return list(self.intent_patterns.keys())
    
    def get_intent_examples(self, intent: str) -> List[str]:
        """Get example phrases for an intent"""
        return self.intent_patterns.get(intent, [])
