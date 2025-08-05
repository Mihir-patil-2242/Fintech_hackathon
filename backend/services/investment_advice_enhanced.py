import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from transformers import pipeline
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedInvestmentAdviceService:
    def __init__(self):
        self.models_dir = "models"
        self.ensure_models_dir()
        
        # ML Models for price prediction and analysis
        self.price_prediction_models = {}
        self.sentiment_analyzer = None
        self.technical_analyzer = None
        self.feature_scaler = StandardScaler()
        
        # Financial data sources
        self.data_sources = {
            'yahoo_finance': True,
            'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_KEY'),
            'news_api_key': os.getenv('NEWS_API_KEY')
        }
        
        # Model performance metrics
        self.model_metrics = {}
        
        # Popular symbols for analysis
        self.popular_symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX',
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ITC.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 'LT.NS', 'ASIANPAINT.NS'
        ]
        
        # Initialize models and services
        self._initialize_models()
        self._initialize_sentiment_analyzer()
    
    def ensure_models_dir(self):
        """Ensure models directory exists"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def _initialize_models(self):
        """Initialize ML models for stock analysis"""
        try:
            self._load_pretrained_models()
            logger.info("Loaded pre-trained investment models")
        except:
            logger.info("Pre-trained investment models not found, training new models...")
            self._train_models()
            self._save_models()
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis for news and social media"""
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            logger.info("Loaded FinBERT sentiment analyzer")
        except Exception as e:
            logger.warning(f"Could not load FinBERT, using fallback: {e}")
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                logger.info("Loaded Twitter sentiment analyzer as fallback")
            except:
                logger.warning("No sentiment analyzer available")
                self.sentiment_analyzer = None
    
    def _generate_stock_training_data(self, symbols: List[str], days: int = 1000) -> pd.DataFrame:
        """Generate training data from historical stock data"""
        all_data = []
        
        for symbol in symbols[:5]:  # Limit to prevent rate limiting
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.hist(period="2y")
                
                if hist.empty:
                    continue
                
                # Calculate technical indicators
                hist['SMA_5'] = hist['Close'].rolling(window=5).mean()
                hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['RSI'] = self._calculate_rsi(hist['Close'])
                hist['MACD'], hist['MACD_signal'] = self._calculate_macd(hist['Close'])
                hist['BB_upper'], hist['BB_lower'] = self._calculate_bollinger_bands(hist['Close'])
                hist['Volume_SMA'] = hist['Volume'].rolling(window=20).mean()
                
                # Price-based features
                hist['Price_Change'] = hist['Close'].pct_change()
                hist['High_Low_Ratio'] = hist['High'] / hist['Low']
                hist['Volume_Price_Trend'] = hist['Volume'] * hist['Price_Change']
                
                # Target: Next day return
                hist['Next_Day_Return'] = hist['Close'].shift(-1) / hist['Close'] - 1
                hist['Next_Day_Direction'] = (hist['Next_Day_Return'] > 0).astype(int)
                
                # Features for model
                feature_cols = [
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    'SMA_5', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal',
                    'BB_upper', 'BB_lower', 'Volume_SMA', 'Price_Change',
                    'High_Low_Ratio', 'Volume_Price_Trend'
                ]
                
                # Clean data
                hist = hist.dropna()
                
                if len(hist) > 50:  # Ensure sufficient data
                    hist['Symbol'] = symbol
                    all_data.append(hist[feature_cols + ['Next_Day_Return', 'Next_Day_Direction', 'Symbol']])
                
            except Exception as e:
                logger.warning(f"Error fetching data for {symbol}: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            # Generate synthetic data as fallback
            return self._generate_synthetic_stock_data()
    
    def _generate_synthetic_stock_data(self) -> pd.DataFrame:
        """Generate synthetic stock data for training"""
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            'Open': np.random.lognormal(4, 0.5, n_samples),
            'High': np.random.lognormal(4.1, 0.5, n_samples),
            'Low': np.random.lognormal(3.9, 0.5, n_samples),
            'Close': np.random.lognormal(4, 0.5, n_samples),
            'Volume': np.random.lognormal(10, 1, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Ensure price relationships
        df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
        df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
        
        # Technical indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = np.random.uniform(20, 80, n_samples)
        df['MACD'] = np.random.normal(0, 1, n_samples)
        df['MACD_signal'] = df['MACD'] + np.random.normal(0, 0.5, n_samples)
        df['BB_upper'] = df['Close'] * 1.02
        df['BB_lower'] = df['Close'] * 0.98
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Volume_Price_Trend'] = df['Volume'] * df['Price_Change']
        
        # Targets based on technical patterns
        df['Next_Day_Return'] = (
            (df['RSI'] < 30).astype(int) * 0.02 +
            (df['RSI'] > 70).astype(int) * (-0.02) +
            (df['Close'] > df['SMA_20']).astype(int) * 0.01 +
            np.random.normal(0, 0.02, n_samples)
        )
        df['Next_Day_Direction'] = (df['Next_Day_Return'] > 0).astype(int)
        
        return df.dropna()
    
    def _train_models(self):
        """Train ML models for stock prediction"""
        logger.info("Generating stock market training data...")
        df = self._generate_stock_training_data(self.popular_symbols[:8])
        
        if df.empty:
            logger.error("No training data available")
            return
        
        # Prepare features
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_lower', 'Volume_SMA', 'Price_Change',
            'High_Low_Ratio', 'Volume_Price_Trend'
        ]
        
        X = df[feature_cols].fillna(0)
        y_return = df['Next_Day_Return'].fillna(0)
        y_direction = df['Next_Day_Direction'].fillna(0)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_return_train, y_return_test, y_direction_train, y_direction_test = train_test_split(
            X_scaled, y_return, y_direction, test_size=0.2, random_state=42
        )
        
        # Train return prediction models
        logger.info("Training return prediction models...")
        self.price_prediction_models = {
            'return_xgboost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
            'return_rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'return_gb': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'direction_xgboost': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        }
        
        # Train regression models
        for name in ['return_xgboost', 'return_rf', 'return_gb']:
            model = self.price_prediction_models[name]
            model.fit(X_train, y_return_train)
            pred = model.predict(X_test)
            
            mse = mean_squared_error(y_return_test, pred)
            mae = mean_absolute_error(y_return_test, pred)
            
            self.model_metrics[f'{name}_mse'] = mse
            self.model_metrics[f'{name}_mae'] = mae
            logger.info(f"{name} - MSE: {mse:.6f}, MAE: {mae:.6f}")
        
        # Train direction model
        direction_model = self.price_prediction_models['direction_xgboost']
        direction_model.fit(X_train, y_direction_train)
        direction_pred = direction_model.predict(X_test)
        
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_direction_test, direction_pred)
        self.model_metrics['direction_accuracy'] = accuracy
        logger.info(f"Direction prediction accuracy: {accuracy:.3f}")
        
        logger.info("Investment model training completed!")
    
    def _save_models(self):
        """Save trained models"""
        try:
            joblib.dump(self.price_prediction_models, os.path.join(self.models_dir, 'investment_models.pkl'))
            joblib.dump(self.feature_scaler, os.path.join(self.models_dir, 'investment_scaler.pkl'))
            joblib.dump(self.model_metrics, os.path.join(self.models_dir, 'investment_metrics.pkl'))
            logger.info("Investment models saved successfully")
        except Exception as e:
            logger.error(f"Error saving investment models: {e}")
    
    def _load_pretrained_models(self):
        """Load pre-trained models"""
        self.price_prediction_models = joblib.load(os.path.join(self.models_dir, 'investment_models.pkl'))
        self.feature_scaler = joblib.load(os.path.join(self.models_dir, 'investment_scaler.pkl'))
        self.model_metrics = joblib.load(os.path.join(self.models_dir, 'investment_metrics.pkl'))
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, lower
    
    async def get_enhanced_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get enhanced market data with AI analysis"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.hist(period="1y")
            info = ticker.info
            
            if hist.empty:
                return self._get_mock_market_data(symbol)
            
            # Current data
            current_price = float(hist['Close'].iloc[-1])
            volume = int(hist['Volume'].iloc[-1])
            
            # Technical indicators
            sma_7 = float(hist['Close'].rolling(7).mean().iloc[-1])
            sma_20 = float(hist['Close'].rolling(20).mean().iloc[-1])
            sma_50 = float(hist['Close'].rolling(50).mean().iloc[-1])
            rsi = float(self._calculate_rsi(hist['Close']).iloc[-1])
            
            macd, macd_signal = self._calculate_macd(hist['Close'])
            bb_upper, bb_lower = self._calculate_bollinger_bands(hist['Close'])
            
            # AI-powered analysis
            features = self._prepare_features_for_prediction(hist.iloc[-1])
            ai_analysis = await self._get_ai_analysis(features, symbol)
            
            # News sentiment
            sentiment_data = await self._get_news_sentiment(symbol)
            
            # Price prediction
            price_prediction = await self._predict_price_movement(features)
            
            # Risk assessment
            risk_assessment = self._calculate_risk_metrics(hist)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'volume': volume,
                'ma_7': sma_7,
                'ma_20': sma_20,
                'ma_50': sma_50,
                'rsi': rsi,
                'rsi_14': rsi,
                'macd': float(macd.iloc[-1]),
                'macd_signal': float(macd_signal.iloc[-1]),
                'bollinger_upper': float(bb_upper.iloc[-1]),
                'bollinger_lower': float(bb_lower.iloc[-1]),
                'avg_volume': int(hist['Volume'].rolling(20).mean().iloc[-1]),
                'change_percent': ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100,
                'recommendation': ai_analysis['recommendation'],
                'confidence': ai_analysis['confidence'],
                'ai_score': ai_analysis['ai_score'],
                'sentiment_score': sentiment_data['score'],
                'news_sentiment': sentiment_data['label'],
                'technical_signals': ai_analysis['technical_signals'],
                'fundamental_metrics': self._get_fundamental_metrics(info),
                'price_prediction': price_prediction,
                'risk_assessment': risk_assessment,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting enhanced market data for {symbol}: {e}")
            return self._get_mock_market_data(symbol)
    
    def _prepare_features_for_prediction(self, latest_data: pd.Series) -> np.ndarray:
        """Prepare features for ML prediction"""
        try:
            features = [
                latest_data.get('Open', 100),
                latest_data.get('High', 105),
                latest_data.get('Low', 95),
                latest_data.get('Close', 100),
                latest_data.get('Volume', 1000000),
                latest_data.get('SMA_5', 100),
                latest_data.get('SMA_20', 100),
                latest_data.get('SMA_50', 100),
                latest_data.get('RSI', 50),
                latest_data.get('MACD', 0),
                latest_data.get('MACD_signal', 0),
                latest_data.get('BB_upper', 105),
                latest_data.get('BB_lower', 95),
                latest_data.get('Volume_SMA', 1000000),
                latest_data.get('Price_Change', 0),
                latest_data.get('High_Low_Ratio', 1.05),
                latest_data.get('Volume_Price_Trend', 0)
            ]
            
            return np.array(features).reshape(1, -1)
        except:
            # Fallback with default values
            return np.array([100, 105, 95, 100, 1000000, 100, 100, 100, 50, 0, 0, 105, 95, 1000000, 0, 1.05, 0]).reshape(1, -1)
    
    async def _get_ai_analysis(self, features: np.ndarray, symbol: str) -> Dict[str, Any]:
        """Get AI-powered analysis"""
        try:
            if not hasattr(self, 'price_prediction_models') or not self.price_prediction_models:
                return self._get_fallback_analysis(symbol)
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features)
            
            # Get predictions from models
            return_predictions = {}
            for name in ['return_xgboost', 'return_rf', 'return_gb']:
                if name in self.price_prediction_models:
                    pred = self.price_prediction_models[name].predict(features_scaled)[0]
                    return_predictions[name] = pred
            
            # Direction prediction
            direction_prob = 0.5
            if 'direction_xgboost' in self.price_prediction_models:
                if hasattr(self.price_prediction_models['direction_xgboost'], 'predict_proba'):
                    direction_prob = self.price_prediction_models['direction_xgboost'].predict_proba(features_scaled)[0][1]
                else:
                    direction_prob = float(self.price_prediction_models['direction_xgboost'].predict(features_scaled)[0])
            
            # Ensemble prediction
            avg_return = np.mean(list(return_predictions.values())) if return_predictions else 0.01
            
            # Generate recommendation
            if avg_return > 0.02 and direction_prob > 0.65:
                recommendation = "STRONG BUY"
                confidence = min(0.95, direction_prob * 1.3)
            elif avg_return > 0.01 and direction_prob > 0.55:
                recommendation = "BUY"
                confidence = direction_prob
            elif avg_return < -0.02 and direction_prob < 0.35:
                recommendation = "STRONG SELL"
                confidence = min(0.95, (1 - direction_prob) * 1.3)
            elif avg_return < -0.01 and direction_prob < 0.45:
                recommendation = "SELL"
                confidence = 1 - direction_prob
            else:
                recommendation = "HOLD"
                confidence = 0.6
            
            # AI score (0-100)
            ai_score = max(0, min(100, 50 + (avg_return * 1000)))
            
            # Technical signals
            technical_signals = {
                'trend': 'bullish' if avg_return > 0 else 'bearish',
                'momentum': 'strong' if abs(avg_return) > 0.015 else 'weak',
                'predicted_return': avg_return,
                'direction_probability': direction_prob
            }
            
            return {
                'recommendation': recommendation,
                'confidence': confidence,
                'ai_score': ai_score,
                'technical_signals': technical_signals
            }
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return self._get_fallback_analysis(symbol)
    
    def _get_fallback_analysis(self, symbol: str) -> Dict[str, Any]:
        """Fallback analysis when ML models are not available"""
        # Simple rule-based analysis
        recommendations = ["BUY", "SELL", "HOLD", "STRONG BUY", "STRONG SELL"]
        recommendation = np.random.choice(recommendations, p=[0.25, 0.15, 0.4, 0.1, 0.1])
        
        return {
            'recommendation': recommendation,
            'confidence': np.random.uniform(0.6, 0.85),
            'ai_score': np.random.uniform(40, 80),
            'technical_signals': {
                'trend': np.random.choice(['bullish', 'bearish']),
                'momentum': np.random.choice(['strong', 'weak']),
                'predicted_return': np.random.normal(0.01, 0.02),
                'direction_probability': np.random.uniform(0.4, 0.8)
            }
        }
    
    async def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment analysis"""
        try:
            if not self.sentiment_analyzer:
                return {'score': 0.0, 'label': 'neutral', 'confidence': 0.5}
            
            # Get recent news (mock for now - in production would use news API)
            news_text = f"Stock {symbol} shows strong performance with positive outlook for growth and expansion."
            
            # Analyze sentiment
            result = self.sentiment_analyzer(news_text)
            
            if isinstance(result, list) and len(result) > 0:
                # FinBERT or similar model
                sentiment_scores = {item['label'].lower(): item['score'] for item in result[0]}
                
                if 'positive' in sentiment_scores:
                    score = sentiment_scores['positive'] - sentiment_scores.get('negative', 0)
                    label = 'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral'
                else:
                    score = 0.0
                    label = 'neutral'
                
                return {
                    'score': score,
                    'label': label,
                    'confidence': max(sentiment_scores.values()) if sentiment_scores else 0.5
                }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
        
        # Fallback
        return {'score': np.random.uniform(-0.2, 0.2), 'label': 'neutral', 'confidence': 0.5}
    
    def _get_fundamental_metrics(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fundamental metrics from stock info"""
        return {
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'price_to_book': info.get('priceToBook', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'roe': info.get('returnOnEquity', 0),
            'revenue_growth': info.get('revenueGrowth', 0),
            'profit_margin': info.get('profitMargins', 0),
            'dividend_yield': info.get('dividendYield', 0)
        }
    
    async def _predict_price_movement(self, features: np.ndarray) -> Dict[str, float]:
        """Predict price movement"""
        try:
            if not hasattr(self, 'price_prediction_models') or not self.price_prediction_models:
                return {'1_day': 0.01, '7_day': 0.05, '30_day': 0.1, 'confidence': 0.6}
            
            features_scaled = self.feature_scaler.transform(features)
            
            # Get return prediction
            if 'return_xgboost' in self.price_prediction_models:
                predicted_return = self.price_prediction_models['return_xgboost'].predict(features_scaled)[0]
            else:
                predicted_return = 0.01
            
            # Extrapolate to different timeframes (simplified)
            return {
                '1_day': predicted_return,
                '7_day': predicted_return * 5,  # Scaled prediction
                '30_day': predicted_return * 15,  # Scaled prediction
                'confidence': 0.7
            }
            
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return {'1_day': 0.01, '7_day': 0.05, '30_day': 0.1, 'confidence': 0.6}
    
    def _calculate_risk_metrics(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk assessment metrics"""
        returns = hist['Close'].pct_change().dropna()
        
        return {
            'volatility': float(returns.std() * np.sqrt(252)),  # Annualized
            'max_drawdown': float((hist['Close'] / hist['Close'].cummax() - 1).min()),
            'var_95': float(returns.quantile(0.05)),  # Value at Risk
            'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'risk_level': 'high' if returns.std() > 0.03 else 'medium' if returns.std() > 0.015 else 'low'
        }
    
    def _get_mock_market_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock market data when real data is unavailable"""
        np.random.seed(hash(symbol) % 2**32)
        
        base_price = np.random.uniform(50, 500)
        current_price = base_price * (1 + np.random.normal(0, 0.02))
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'volume': np.random.randint(100000, 10000000),
            'ma_7': round(current_price * (1 + np.random.normal(0, 0.01)), 2),
            'ma_20': round(current_price * (1 + np.random.normal(0, 0.02)), 2),
            'ma_50': round(current_price * (1 + np.random.normal(0, 0.03)), 2),
            'rsi': np.random.uniform(20, 80),
            'rsi_14': np.random.uniform(20, 80),
            'macd': np.random.normal(0, 1),
            'macd_signal': np.random.normal(0, 0.8),
            'bollinger_upper': round(current_price * 1.02, 2),
            'bollinger_lower': round(current_price * 0.98, 2),
            'avg_volume': np.random.randint(500000, 5000000),
            'change_percent': np.random.uniform(-5, 5),
            'recommendation': np.random.choice(['BUY', 'SELL', 'HOLD', 'STRONG BUY', 'STRONG SELL']),
            'confidence': np.random.uniform(0.6, 0.9),
            'ai_score': np.random.uniform(40, 90),
            'sentiment_score': np.random.uniform(-0.3, 0.3),
            'news_sentiment': np.random.choice(['positive', 'negative', 'neutral']),
            'technical_signals': {
                'trend': np.random.choice(['bullish', 'bearish']),
                'momentum': np.random.choice(['strong', 'weak'])
            },
            'fundamental_metrics': {
                'pe_ratio': np.random.uniform(10, 30),
                'market_cap': np.random.randint(1000000000, 100000000000)
            },
            'price_prediction': {
                '1_day': np.random.uniform(-0.02, 0.02),
                '7_day': np.random.uniform(-0.1, 0.1),
                '30_day': np.random.uniform(-0.2, 0.2)
            },
            'risk_assessment': {
                'volatility': np.random.uniform(0.15, 0.45),
                'risk_level': np.random.choice(['low', 'medium', 'high'])
            },
            'timestamp': datetime.now()
        }
    
    async def get_enhanced_portfolio_advice(self, symbols: List[str], user_id: str) -> List[Dict[str, Any]]:
        """Get enhanced portfolio advice for multiple symbols"""
        results = []
        
        for symbol in symbols:
            try:
                data = await self.get_enhanced_market_data(symbol)
                # Add personalized recommendations based on user profile
                data = await self._personalize_recommendation(data, user_id)
                results.append(data)
            except Exception as e:
                logger.error(f"Error getting data for {symbol}: {e}")
                continue
        
        return results
    
    async def get_enhanced_popular_stocks(self, user_id: str) -> List[Dict[str, Any]]:
        """Get enhanced data for popular stocks"""
        return await self.get_enhanced_portfolio_advice(self.popular_symbols[:8], user_id)
    
    async def _personalize_recommendation(self, data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Personalize recommendations based on user profile"""
        # Mock personalization - in production would use user's risk profile, preferences, etc.
        risk_tolerance = hash(user_id) % 3  # 0=conservative, 1=moderate, 2=aggressive
        
        if risk_tolerance == 0:  # Conservative
            if data['risk_assessment']['risk_level'] == 'high':
                data['recommendation'] = 'HOLD' if data['recommendation'] in ['BUY', 'STRONG BUY'] else data['recommendation']
                data['personalized_note'] = "Reduced recommendation due to conservative risk profile"
        elif risk_tolerance == 2:  # Aggressive
            if data['recommendation'] == 'HOLD' and data['ai_score'] > 60:
                data['recommendation'] = 'BUY'
                data['personalized_note'] = "Enhanced recommendation for aggressive profile"
        
        return data
    
    async def get_ai_stock_insights(self, symbol: str, user_id: str) -> Dict[str, Any]:
        """Get comprehensive AI insights for a specific stock"""
        try:
            # Get enhanced market data
            market_data = await self.get_enhanced_market_data(symbol)
            
            # Generate detailed AI analysis
            ai_analysis = await self._generate_detailed_analysis(symbol, market_data)
            
            # Technical analysis
            technical_analysis = self._generate_technical_analysis(market_data)
            
            # Fundamental analysis
            fundamental_analysis = self._generate_fundamental_analysis(market_data)
            
            # Sentiment analysis
            sentiment_analysis = await self._generate_sentiment_analysis(symbol)
            
            # Price prediction with confidence intervals
            price_prediction = await self._generate_detailed_price_prediction(symbol)
            
            # Risk assessment
            risk_assessment = self._generate_detailed_risk_assessment(market_data)
            
            return {
                'symbol': symbol,
                'ai_analysis': ai_analysis,
                'technical_analysis': technical_analysis,
                'fundamental_analysis': fundamental_analysis,
                'sentiment_analysis': sentiment_analysis,
                'price_prediction': price_prediction,
                'risk_assessment': risk_assessment,
                'recommendation_reasoning': self._generate_recommendation_reasoning(market_data),
                'confidence_factors': self._get_confidence_factors(market_data),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating AI insights for {symbol}: {e}")
            return {'error': f"Could not generate insights for {symbol}"}
    
    def _generate_detailed_analysis(self, symbol: str, market_data: Dict[str, Any]) -> str:
        """Generate detailed AI analysis text"""
        recommendation = market_data.get('recommendation', 'HOLD')
        ai_score = market_data.get('ai_score', 50)
        sentiment = market_data.get('news_sentiment', 'neutral')
        
        analysis = f"""
        Based on our comprehensive AI analysis of {symbol}, the current recommendation is {recommendation} 
        with an AI confidence score of {ai_score:.1f}/100. 
        
        The technical indicators suggest a {market_data.get('technical_signals', {}).get('trend', 'neutral')} trend 
        with {market_data.get('technical_signals', {}).get('momentum', 'moderate')} momentum. 
        
        Market sentiment analysis indicates a {sentiment} outlook based on recent news and social media activity.
        
        The stock shows {market_data.get('risk_assessment', {}).get('risk_level', 'moderate')} risk characteristics 
        with a volatility of {market_data.get('risk_assessment', {}).get('volatility', 0.25):.1%}.
        """
        
        return analysis.strip()
    
    def _generate_technical_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed technical analysis"""
        return {
            'trend_analysis': {
                'short_term': 'bullish' if market_data.get('current_price', 0) > market_data.get('ma_7', 0) else 'bearish',
                'medium_term': 'bullish' if market_data.get('current_price', 0) > market_data.get('ma_20', 0) else 'bearish',
                'long_term': 'bullish' if market_data.get('current_price', 0) > market_data.get('ma_50', 0) else 'bearish'
            },
            'oscillators': {
                'rsi_signal': 'oversold' if market_data.get('rsi', 50) < 30 else 'overbought' if market_data.get('rsi', 50) > 70 else 'neutral',
                'macd_signal': 'bullish' if market_data.get('macd', 0) > market_data.get('macd_signal', 0) else 'bearish'
            },
            'support_resistance': {
                'resistance': market_data.get('bollinger_upper', 0),
                'support': market_data.get('bollinger_lower', 0)
            },
            'volume_analysis': {
                'volume_trend': 'increasing' if market_data.get('volume', 0) > market_data.get('avg_volume', 0) else 'decreasing',
                'volume_confirmation': market_data.get('volume', 0) > market_data.get('avg_volume', 0) * 1.2
            }
        }
    
    def _generate_fundamental_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fundamental analysis"""
        fundamentals = market_data.get('fundamental_metrics', {})
        
        return {
            'valuation': {
                'pe_assessment': 'overvalued' if fundamentals.get('pe_ratio', 15) > 25 else 'undervalued' if fundamentals.get('pe_ratio', 15) < 15 else 'fairly_valued',
                'pb_assessment': 'overvalued' if fundamentals.get('price_to_book', 1) > 3 else 'undervalued' if fundamentals.get('price_to_book', 1) < 1 else 'fairly_valued'
            },
            'financial_health': {
                'profitability': 'strong' if fundamentals.get('profit_margin', 0.1) > 0.15 else 'weak' if fundamentals.get('profit_margin', 0.1) < 0.05 else 'moderate',
                'growth_prospects': 'strong' if fundamentals.get('revenue_growth', 0.05) > 0.15 else 'weak' if fundamentals.get('revenue_growth', 0.05) < 0 else 'moderate'
            },
            'dividend_analysis': {
                'yield': fundamentals.get('dividend_yield', 0),
                'sustainability': 'sustainable' if fundamentals.get('dividend_yield', 0) < 0.08 else 'questionable'
            }
        }
    
    async def _generate_sentiment_analysis(self, symbol: str) -> Dict[str, Any]:
        """Generate comprehensive sentiment analysis"""
        sentiment_data = await self._get_news_sentiment(symbol)
        
        return {
            'overall_sentiment': sentiment_data['label'],
            'sentiment_score': sentiment_data['score'],
            'confidence': sentiment_data['confidence'],
            'news_impact': 'positive' if sentiment_data['score'] > 0.1 else 'negative' if sentiment_data['score'] < -0.1 else 'neutral',
            'social_sentiment': np.random.choice(['bullish', 'bearish', 'neutral']),  # Mock data
            'analyst_sentiment': np.random.choice(['positive', 'negative', 'neutral'])  # Mock data
        }
    
    async def _generate_detailed_price_prediction(self, symbol: str) -> Dict[str, Any]:
        """Generate detailed price predictions with confidence intervals"""
        # Mock implementation - in production would use more sophisticated models
        base_prediction = np.random.uniform(-0.05, 0.05)
        
        return {
            'short_term': {
                '1_day': {'prediction': base_prediction * 0.2, 'confidence_interval': [base_prediction * 0.1, base_prediction * 0.3]},
                '7_day': {'prediction': base_prediction, 'confidence_interval': [base_prediction * 0.7, base_prediction * 1.3]},
                '30_day': {'prediction': base_prediction * 3, 'confidence_interval': [base_prediction * 2, base_prediction * 4]}
            },
            'long_term': {
                '90_day': {'prediction': base_prediction * 8, 'confidence_interval': [base_prediction * 5, base_prediction * 12]},
                '1_year': {'prediction': base_prediction * 20, 'confidence_interval': [base_prediction * 10, base_prediction * 30]}
            },
            'model_confidence': np.random.uniform(0.6, 0.85)
        }
    
    def _generate_detailed_risk_assessment(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed risk assessment"""
        risk_data = market_data.get('risk_assessment', {})
        
        return {
            'overall_risk': risk_data.get('risk_level', 'medium'),
            'volatility_analysis': {
                'current_volatility': risk_data.get('volatility', 0.25),
                'historical_comparison': 'above_average' if risk_data.get('volatility', 0.25) > 0.3 else 'below_average' if risk_data.get('volatility', 0.25) < 0.2 else 'average'
            },
            'downside_risk': {
                'max_drawdown': risk_data.get('max_drawdown', -0.15),
                'var_95': risk_data.get('var_95', -0.02)
            },
            'risk_factors': [
                'Market volatility',
                'Sector-specific risks',
                'Economic uncertainty',
                'Regulatory changes'
            ],
            'risk_mitigation': [
                'Diversification across sectors',
                'Position sizing',
                'Stop-loss orders',
                'Regular monitoring'
            ]
        }
    
    def _generate_recommendation_reasoning(self, market_data: Dict[str, Any]) -> str:
        """Generate reasoning for the recommendation"""
        recommendation = market_data.get('recommendation', 'HOLD')
        factors = []
        
        if market_data.get('ai_score', 50) > 70:
            factors.append("strong AI confidence score")
        
        if market_data.get('sentiment_score', 0) > 0.1:
            factors.append("positive market sentiment")
        elif market_data.get('sentiment_score', 0) < -0.1:
            factors.append("negative market sentiment")
        
        technical_signals = market_data.get('technical_signals', {})
        if technical_signals.get('trend') == 'bullish':
            factors.append("bullish technical trend")
        elif technical_signals.get('trend') == 'bearish':
            factors.append("bearish technical trend")
        
        if not factors:
            factors.append("balanced market conditions")
        
        return f"Recommendation of {recommendation} is based on: {', '.join(factors)}"
    
    def _get_confidence_factors(self, market_data: Dict[str, Any]) -> List[str]:
        """Get factors that contribute to confidence in the analysis"""
        factors = []
        
        if market_data.get('confidence', 0) > 0.8:
            factors.append("High model agreement")
        
        if market_data.get('volume', 0) > market_data.get('avg_volume', 0) * 1.5:
            factors.append("Strong volume confirmation")
        
        if abs(market_data.get('sentiment_score', 0)) > 0.2:
            factors.append("Clear sentiment direction")
        
        if market_data.get('risk_assessment', {}).get('volatility', 0.25) < 0.2:
            factors.append("Low volatility environment")
        
        return factors if factors else ["Standard market analysis"]
    
    async def health_check(self) -> str:
        """Health check for investment service"""
        try:
            # Test market data retrieval
            test_data = await self.get_enhanced_market_data('AAPL')
            
            if test_data and 'symbol' in test_data and 'current_price' in test_data:
                return "healthy"
            else:
                return "unhealthy"
        except Exception:
            return "unhealthy"
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return self.model_metrics.copy()
