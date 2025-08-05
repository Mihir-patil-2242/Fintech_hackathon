from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class LoanRequest(BaseModel):
    user_id: str
    income: float
    loan_amount: float
    credit_history_score: int  # 1-10 scale
    employment_years: int
    existing_debts: float
    purpose: str

class LoanResponse(BaseModel):
    loan_id: str
    approved: bool
    credit_score: int
    reason: str
    blockchain_tx_id: Optional[str] = None

class FraudCheckRequest(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    merchant: str
    location: str
    time_of_day: int  # 0-23 hours

class FraudResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    blockchain_tx_id: Optional[str] = None

class ComplianceCheckRequest(BaseModel):
    user_id: str
    transaction_amount: float
    kyc_completed: bool
    source_of_funds: str

class ComplianceResponse(BaseModel):
    user_id: str
    compliant: bool
    violations: List[str]
    blockchain_tx_id: Optional[str] = None

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    confidence: float

class MarketData(BaseModel):
    symbol: str
    current_price: float
    ma_7: float
    rsi: float
    recommendation: str  # BUY, SELL, HOLD
    timestamp: datetime
