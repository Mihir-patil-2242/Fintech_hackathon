import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

class InvestmentAdviceService:
    def __init__(self):
        # Popular Indian and international stocks for demo
        self.popular_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'
        ]
    
    def get_market_data(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """Fetch market data and calculate technical indicators"""
        try:
            # Fetch data from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.hist(period=period)
            
            if hist.empty:
                return self._get_mock_data(symbol)
            
            # Calculate technical indicators
            current_price = float(hist['Close'].iloc[-1])
            ma_7 = float(hist['Close'].rolling(window=min(7, len(hist))).mean().iloc[-1])
            rsi = self._calculate_rsi(hist['Close'])
            
            # Generate recommendation
            recommendation = self._generate_recommendation(current_price, ma_7, rsi)
            
            return {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'ma_7': round(ma_7, 2),
                'rsi': round(rsi, 2),
                'recommendation': recommendation,
                'timestamp': datetime.now(),
                'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                'change_percent': round(((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100, 2) if len(hist) > 1 else 0
            }
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return self._get_mock_data(symbol)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < window:
            window = len(prices) - 1
            
        if window <= 1:
            return 50.0  # Neutral RSI
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
    
    def _generate_recommendation(self, current_price: float, ma_7: float, rsi: float) -> str:
        """Generate buy/sell/hold recommendation based on technical analysis"""
        
        # RSI-based signals
        rsi_signal = ""
        if rsi < 30:
            rsi_signal = "BUY"  # Oversold
        elif rsi > 70:
            rsi_signal = "SELL"  # Overbought
        else:
            rsi_signal = "HOLD"
        
        # Moving average signal
        ma_signal = ""
        if current_price > ma_7 * 1.02:  # 2% above MA
            ma_signal = "BUY"
        elif current_price < ma_7 * 0.98:  # 2% below MA
            ma_signal = "SELL"
        else:
            ma_signal = "HOLD"
        
        # Combine signals (RSI weighted more heavily)
        if rsi_signal == "BUY" and ma_signal in ["BUY", "HOLD"]:
            return "STRONG BUY"
        elif rsi_signal == "BUY":
            return "BUY"
        elif rsi_signal == "SELL" and ma_signal in ["SELL", "HOLD"]:
            return "STRONG SELL"
        elif rsi_signal == "SELL":
            return "SELL"
        else:
            return "HOLD"
    
    def _get_mock_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock data when yfinance fails"""
        np.random.seed(hash(symbol) % 2**32)  # Consistent random data per symbol
        
        base_price = np.random.uniform(100, 2000)
        current_price = base_price * (1 + np.random.normal(0, 0.02))
        ma_7 = base_price * (1 + np.random.normal(0, 0.01))
        rsi = np.random.uniform(20, 80)
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'ma_7': round(ma_7, 2),
            'rsi': round(rsi, 2),
            'recommendation': self._generate_recommendation(current_price, ma_7, rsi),
            'timestamp': datetime.now(),
            'volume': np.random.randint(100000, 1000000),
            'change_percent': round(np.random.uniform(-5, 5), 2)
        }
    
    def get_portfolio_advice(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Get advice for multiple symbols"""
        results = []
        for symbol in symbols:
            data = self.get_market_data(symbol)
            results.append(data)
        return results
    
    def get_popular_stocks(self) -> List[Dict[str, Any]]:
        """Get data for popular stocks"""
        return self.get_portfolio_advice(self.popular_symbols[:5])  # Limit for demo
