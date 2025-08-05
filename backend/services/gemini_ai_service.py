import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import google.generativeai as genai
from datetime import datetime
import asyncio
import aiohttp
import numpy as np
from transformers import pipeline
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiAIService:
    def __init__(self):
        # Initialize Gemini AI
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("Gemini AI initialized successfully")
        else:
            self.model = None
            logger.warning("Gemini API key not found, using fallback responses")
        
        # Initialize sentiment analysis model
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            logger.info("FinBERT sentiment analyzer loaded")
        except Exception as e:
            logger.warning(f"Could not load FinBERT: {e}")
            self.sentiment_analyzer = None
    
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate AI response using Gemini"""
        try:
            if not self.model:
                return self._fallback_response(prompt)
            
            # Enhance prompt with context
            enhanced_prompt = self._enhance_prompt(prompt, context)
            
            # Generate response
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.model.generate_content(enhanced_prompt)
            )
            
            return {
                "response": response.text,
                "confidence": 0.85,  # Gemini typically has high confidence
                "model": "gemini-2.0-flash-exp",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._fallback_response(prompt)
    
    def _enhance_prompt(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Enhance prompt with context and instructions"""
        system_prompt = """You are an expert financial AI assistant. You provide accurate, helpful, and professional financial advice. Always be clear about risks and limitations. If you're unsure about something, say so rather than guessing.

Key principles:
- Be accurate and factual
- Explain financial concepts clearly
- Always mention risks and disclaimers
- Be helpful but conservative with advice
- Use proper financial terminology
- Provide actionable insights when possible

"""
        
        if context:
            context_str = f"Context: {json.dumps(context, indent=2)}\n\n"
            return system_prompt + context_str + f"User question: {prompt}"
        
        return system_prompt + f"User question: {prompt}"
    
    def _fallback_response(self, prompt: str) -> Dict[str, Any]:
        """Fallback response when Gemini is not available"""
        fallback_responses = {
            "loan": "I can help you understand loan requirements. Generally, lenders look at your credit score, income, debt-to-income ratio, and employment history. Would you like me to explain any of these factors?",
            "credit": "Credit scores typically range from 300-850. Factors that affect your score include payment history (35%), credit utilization (30%), length of credit history (15%), new credit (10%), and types of credit (10%).",
            "investment": "Investment decisions should be based on your risk tolerance, time horizon, and financial goals. Diversification is key to managing risk. Always do your research and consider consulting with a financial advisor.",
            "fraud": "If you suspect fraud, act quickly: contact your bank immediately, change your passwords, review your statements, and consider freezing your credit. Document everything and report to authorities if needed.",
            "default": "I'm here to help with your financial questions. I can assist with loans, credit scores, investments, fraud protection, and general financial planning. What specific area would you like to discuss?"
        }
        
        # Simple keyword matching for fallback
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ['loan', 'borrow', 'mortgage']):
            response_text = fallback_responses['loan']
        elif any(word in prompt_lower for word in ['credit', 'score', 'rating']):
            response_text = fallback_responses['credit']
        elif any(word in prompt_lower for word in ['invest', 'stock', 'market', 'portfolio']):
            response_text = fallback_responses['investment']
        elif any(word in prompt_lower for word in ['fraud', 'suspicious', 'scam', 'hack']):
            response_text = fallback_responses['fraud']
        else:
            response_text = fallback_responses['default']
        
        return {
            "response": response_text,
            "confidence": 0.6,
            "model": "fallback",
            "timestamp": datetime.now().isoformat()
        }
    
    async def analyze_financial_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of financial text"""
        try:
            if self.sentiment_analyzer:
                result = self.sentiment_analyzer(text)
                if isinstance(result, list) and len(result) > 0:
                    sentiment_scores = {item['label'].lower(): item['score'] for item in result[0]}
                    
                    # Determine overall sentiment
                    if 'positive' in sentiment_scores and 'negative' in sentiment_scores:
                        score = sentiment_scores['positive'] - sentiment_scores['negative']
                        if score > 0.1:
                            label = 'positive'
                        elif score < -0.1:
                            label = 'negative'
                        else:
                            label = 'neutral'
                    else:
                        label = 'neutral'
                        score = 0.0
                    
                    return {
                        'sentiment': label,
                        'score': float(score),
                        'confidence': max(sentiment_scores.values()) if sentiment_scores else 0.5,
                        'details': sentiment_scores
                    }
            
            # Fallback sentiment analysis
            return await self._fallback_sentiment_analysis(text)
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return await self._fallback_sentiment_analysis(text)
    
    async def _fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'positive', 'bullish', 'up', 'gain', 'profit', 'success']
        negative_words = ['bad', 'terrible', 'negative', 'bearish', 'down', 'loss', 'decline', 'fail']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = 0.1 + (positive_count - negative_count) * 0.1
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = -0.1 - (negative_count - positive_count) * 0.1
        else:
            sentiment = 'neutral'
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': float(np.clip(score, -1.0, 1.0)),
            'confidence': 0.6,
            'details': {'method': 'rule-based'}
        }
    
    async def health_check(self) -> str:
        """Health check for AI service"""
        try:
            if self.model:
                # Test Gemini API
                test_response = await self.generate_response("Test message")
                if test_response and 'response' in test_response:
                    return "healthy"
            
            # Test fallback
            fallback_response = self._fallback_response("test")
            if fallback_response and 'response' in fallback_response:
                return "healthy"
            
            return "unhealthy"
            
        except Exception as e:
            logger.error(f"AI service health check failed: {e}")
            return "unhealthy"
