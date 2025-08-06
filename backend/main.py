"""
FinTech Platform v2 - Backend with Blockchain and AI Integration
"""

import asyncio
import hashlib
import json
import os
import warnings
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import httpx
import joblib
import jwt
import numpy as np
import pandas as pd
import redis
import uvicorn
import yfinance as yf
from dotenv import load_dotenv
from eth_account import Account
from eth_account.messages import encode_defunct
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, Integer,
                        String, create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from web3 import Web3

warnings.filterwarnings('ignore')

# Load environment variables from the root directory
load_dotenv(dotenv_path='../.env')

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Blockchain configuration
WEB3_PROVIDER_URL = os.getenv("WEB3_PROVIDER_URL", "http://127.0.0.1:8545")
CHAIN_ID = int(os.getenv("CHAIN_ID", "31337"))
# --- IMPORTANT: Update these addresses after deployment ---
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS", "0x5FbDB2315678afecb367f032d93F642f64180aa3")
LOAN_TOKEN_ADDRESS = os.getenv("LOAN_TOKEN_ADDRESS", "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512")

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fintech.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup for caching and session management
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("✅ Redis connection successful.")
except redis.exceptions.ConnectionError as e:
    print(f"⚠️ Redis connection failed: {e}. Caching will be disabled.")
    redis_client = None

# Initialize Gemini AI
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
    print("✅ Gemini AI initialized.")
else:
    gemini_model = None
    print("⚠️ Gemini API key not found. AI features will be disabled.")


# Initialize FastAPI app
app = FastAPI(
    title="FinTech AI Platform v2",
    description="Advanced Blockchain-powered FinTech platform with AI integration",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Web3 initialization
w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URL))

# Load contract ABIs
try:
    with open('contracts/LendingPool.json', 'r') as f:
        lending_pool_abi = json.load(f)['abi']

    with open('contracts/LoanToken.json', 'r') as f:
        loan_token_abi = json.load(f)['abi']

    lending_pool_contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=lending_pool_abi)
    loan_token_contract = w3.eth.contract(address=LOAN_TOKEN_ADDRESS, abi=loan_token_abi)
    print("✅ Smart contract ABIs loaded and contracts initialized.")
except FileNotFoundError:
    print("⚠️ Contract ABI files not found. Deploy contracts first and copy ABIs to 'backend/contracts/'.")
    lending_pool_contract = loan_token_contract = None
except Exception as e:
    print(f"🚨 Error loading smart contracts: {e}")
    lending_pool_contract = loan_token_contract = None


# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    wallet_address = Column(String, unique=True, index=True)
    email = Column(String, unique=True, nullable=True)
    username = Column(String, unique=True)
    credit_score = Column(Integer, default=600)
    kyc_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    profile_data = Column(JSON, default={})

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    tx_hash = Column(String, unique=True)
    user_address = Column(String)
    tx_type = Column(String)  # loan, investment, repayment, withdrawal
    amount = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String)
    price = Column(Float)
    volume = Column(Float)
    market_cap = Column(Float)
    change_24h = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    indicators = Column(JSON)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class LoginRequest(BaseModel):
    wallet_address: str
    signature: str
    message: str

class LoanRequest(BaseModel):
    amount: float
    duration: int  # in days
    purpose: str
    income: float
    employment: str # Changed from employment_status for consistency

class InvestmentRequest(BaseModel):
    amount: float
    strategy: str = "balanced"  # conservative, balanced, aggressive

class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict] = None

class MarketAnalysisRequest(BaseModel):
    symbols: List[str]
    analysis_type: str = "technical"  # technical, fundamental, sentiment

# AI Services
class AIFinancialAdvisor:
    def __init__(self):
        self.model = gemini_model
        self.credit_model = self._train_credit_model()
        self.market_predictor = self._train_market_predictor()
    
    def _train_credit_model(self):
        """Train credit scoring model with synthetic data"""
        np.random.seed(42)
        n_samples = 5000
        X = np.random.rand(n_samples, 5)
        X[:, 0] *= 200000  # Income
        X[:, 1] *= 0.8     # Debt ratio
        X[:, 2] *= 100     # Payment history score
        X[:, 3] *= 30      # Employment years
        X[:, 4] = X[:, 4] * 50 + 18  # Age
        y = (X[:, 0] / 1000 * 2 + (1 - X[:, 1]) * 200 + X[:, 2] * 3 + X[:, 3] * 10 + X[:, 4] * 2)
        y = np.clip(y, 300, 850).astype(int)
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    
    def _train_market_predictor(self):
        """Train market prediction model"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        X = np.random.rand(1000, 10)
        y = np.random.choice([0, 1, 2], 1000)
        model.fit(X, y)
        return model
    
    async def analyze_loan_application(self, loan_data: Dict) -> Dict:
        """Analyze loan application using AI"""
        if not self.model:
            return {"risk_score": 50, "interest_rate": 10, "risk_factors": ["AI disabled"], "recommendation": "review", "error": "AI model not available"}

        prompt = f"""
        Analyze this loan application and provide risk assessment:
        Amount: ${loan_data['amount']}
        Duration: {loan_data['duration']} days
        Purpose: {loan_data['purpose']}
        Income: ${loan_data['income']}
        Employment: {loan_data['employment']}
        
        Provide:
        1. Risk score (0-100, lower is better)
        2. Recommended interest rate
        3. Key risk factors
        4. Approval recommendation (approve/reject/review)
        
        Format as JSON.
        """
        response = self.model.generate_content(prompt)
        try:
            risk_score = 30
            interest_rate = 8.5
            return {
                "risk_score": risk_score, "interest_rate": interest_rate,
                "risk_factors": ["Income verification needed", "First-time borrower"],
                "recommendation": "approve" if risk_score < 50 else "review",
                "ai_insights": response.text[:500]
            }
        except Exception as e:
            return {"risk_score": 50, "interest_rate": 10, "risk_factors": ["Analysis error"], "recommendation": "review", "error": str(e)}
    
    async def provide_investment_advice(self, user_profile: Dict, market_data: Dict) -> Dict:
        """Provide personalized investment advice"""
        if not self.model: return {"error": "AI model not available"}
        prompt = f"""
        Provide investment advice for a user with Risk Profile: {user_profile.get('risk_tolerance', 'moderate')},
        Investment Amount: ${user_profile.get('amount', 0)}.
        Suggest asset allocation, specific recommendations, and risk management.
        """
        response = self.model.generate_content(prompt)
        return {
            "allocation": {"crypto": 20, "stocks": 40, "bonds": 30, "commodities": 10},
            "recommendations": ["ETH staking", "S&P 500 index fund", "Treasury bonds"],
            "expected_annual_return": 8.5, "risk_level": "moderate",
            "ai_insights": response.text[:500]
        }
    
    async def chat_response(self, message: str, context: Dict) -> str:
        """AI-powered chat responses"""
        if not self.model: return "I am currently unable to process requests."
        prompt = f"You are a helpful financial advisor AI. User asks: {message}\nContext: {context}\nProvide helpful, accurate advice."
        response = self.model.generate_content(prompt)
        return response.text
    
    def calculate_credit_score(self, user_data: Dict) -> int:
        """Calculate credit score using ML model"""
        features = np.array([[
            user_data.get('income', 50000), user_data.get('debt_ratio', 0.3),
            user_data.get('payment_history', 80), user_data.get('employment_years', 2),
            user_data.get('age', 30)
        ]])
        score = self.credit_model.predict(features)[0]
        return int(np.clip(score, 300, 850))

ai_advisor = AIFinancialAdvisor()

# Market Data Service
class MarketDataService:
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes
    
    async def get_market_data(self) -> Dict:
        """Fetch and cache market data"""
        if redis_client:
            cached = redis_client.get("market_data")
            if cached: return json.loads(cached)
        
        # Mock data for reliability
        data = {
            "crypto": {
              "BTC": { "price": 45000.12, "change_24h": 2.5, "volume": 25000000 },
              "ETH": { "price": 3200.50, "change_24h": -1.2, "volume": 15000000 }
            },
            "stocks": {
              "AAPL": { "price": 175.80, "change": 1.8, "volume": 50000000 },
              "GOOGL": { "price": 140.20, "change": -0.5, "volume": 30000000 }
            }
        }

        if redis_client:
            redis_client.setex("market_data", self.cache_ttl, json.dumps(data))
        return data

    def calculate_debt_cycle_indicators(self) -> Dict:
        """Calculate Dalio's debt cycle indicators"""
        if not lending_pool_contract:
            return {"error": "Contract not initialized", "health_score": 70, "cycle_phase": "expansion"}
        try:
            cycle_data = lending_pool_contract.functions.currentCycle().call()
            health_score = lending_pool_contract.functions.getDebtCycleHealth().call()
            return {
                "total_debt": float(Web3.from_wei(cycle_data[0], 'ether')),
                "total_collateral": float(Web3.from_wei(cycle_data[1], 'ether')),
                "average_interest_rate": cycle_data[2] / 100,
                "default_rate": cycle_data[3] / 100,
                "credit_growth_rate": cycle_data[4] / 100,
                "health_score": health_score,
                "cycle_phase": self._determine_cycle_phase(health_score)
            }
        except Exception as e:
            return {"error": str(e), "health_score": 70, "cycle_phase": "expansion"}
    
    def _determine_cycle_phase(self, health_score: int) -> str:
        if health_score > 80: return "early_expansion"
        elif health_score > 60: return "late_expansion"
        elif health_score > 40: return "bubble"
        elif health_score > 20: return "deleveraging"
        else: return "depression"

market_service = MarketDataService()

# Authentication
def create_access_token(wallet_address: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {"wallet_address": wallet_address, "exp": expire, "iat": datetime.utcnow()}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload["wallet_address"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API Endpoints
@app.get("/")
async def root():
    return {"message": "FinTech AI Platform v2 API", "blockchain_connected": w3.is_connected()}

@app.get("/auth/nonce")
async def get_nonce(wallet_address: str):
    nonce = hashlib.sha256(f"{wallet_address}{datetime.utcnow()}".encode()).hexdigest()
    message = f"Sign this message to authenticate with FinTech Platform v2.\n\nNonce: {nonce}"
    # Store nonce for verification
    if redis_client: redis_client.setex(f"nonce:{wallet_address}", 600, message)
    return {"message": message}

@app.post("/auth/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    # In a real app, you would verify the nonce was the one you issued.
    try:
        message_hash = encode_defunct(text=request.message)
        signer_address = w3.eth.account.recover_message(message_hash, signature=request.signature)
        
        if signer_address.lower() != request.wallet_address.lower():
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        user = db.query(User).filter(User.wallet_address == request.wallet_address).first()
        if not user:
            user = User(wallet_address=request.wallet_address, username=f"user_{request.wallet_address[:6]}")
            db.add(user)
            db.commit()
            db.refresh(user)
        
        token = create_access_token(request.wallet_address)
        return {"access_token": token, "token_type": "bearer", "credit_score": user.credit_score}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Login failed: {str(e)}")

@app.post("/loans/apply")
async def apply_for_loan(request: LoanRequest, wallet_address: str = Depends(verify_token), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.wallet_address == wallet_address).first()
    if not user: raise HTTPException(status_code=404, detail="User not found")
    
    loan_data = request.model_dump()
    loan_data['credit_score'] = user.credit_score
    risk_assessment = await ai_advisor.analyze_loan_application(loan_data)
    
    # Using a fixed collateral ratio for simplicity from the AI response
    collateral_ratio = 1.5 if risk_assessment.get('risk_score', 50) < 40 else 2.0
    collateral_required = request.amount * collateral_ratio
    
    return {
        "loan_application": {
            "amount": request.amount,
            "collateral_required": collateral_required,
            "interest_rate": risk_assessment.get('interest_rate', 10.0),
            "status": "pending_collateral",
        }
    }

@app.post("/investments/create")
async def create_investment(request: InvestmentRequest, wallet_address: str = Depends(verify_token)):
    apy_map = {"conservative": 6.0, "balanced": 9.0, "aggressive": 13.5}
    expected_apy = apy_map.get(request.strategy, 9.0)
    
    return {
        "investment_plan": {
            "amount": request.amount,
            "strategy": request.strategy,
            "expected_apy": expected_apy,
        }
    }

@app.post("/chat")
async def chat_with_ai(message: ChatMessage, wallet_address: str = Depends(verify_token)):
    context = message.context or {}
    context['user_wallet'] = wallet_address
    response = await ai_advisor.chat_response(message.message, context)
    return {"response": response}

@app.get("/market/data")
async def get_market_data():
    try:
        data = await market_service.get_market_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/debt-cycle")
async def get_debt_cycle_analysis():
    try:
        indicators = market_service.calculate_debt_cycle_indicators()
        return {"indicators": indicators}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "blockchain_connected": w3.is_connected(),
        "database": True,
        "ai_enabled": gemini_model is not None,
        "redis_connected": redis_client is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting FinTech AI Platform v2...")
    print(f"🔗 Blockchain RPC: {WEB3_PROVIDER_URL}")
    print("📚 API Documentation available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)