from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

# ============================================================================
# AUTHENTICATION MODELS
# ============================================================================

class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone_number: Optional[str] = None
    date_of_birth: Optional[str] = None
    address: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    user_id: str
    email: str
    full_name: str
    created_at: datetime
    blockchain_tx_id: Optional[str] = None
    is_verified: bool = False

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    email: str
    expires_in: int = 3600

# ============================================================================
# ENHANCED LOAN MODELS
# ============================================================================

class LoanRequest(BaseModel):
    income: float
    loan_amount: float
    credit_history_score: int  # 1-10 scale
    employment_years: int
    existing_debts: float
    purpose: str
    employment_type: Optional[str] = "full_time"
    monthly_expenses: Optional[float] = 0
    assets_value: Optional[float] = 0
    co_applicant: Optional[bool] = False

class LoanResponse(BaseModel):
    loan_id: str
    approved: bool
    credit_score: int
    reason: str
    blockchain_tx_id: Optional[str] = None
    confidence: Optional[float] = None
    risk_factors: Optional[List[str]] = []
    fraud_probability: Optional[float] = None
    recommended_amount: Optional[float] = None
    interest_rate: Optional[float] = None
    loan_term_months: Optional[int] = None

# ============================================================================
# ENHANCED FRAUD DETECTION MODELS
# ============================================================================

class FraudCheckRequest(BaseModel):
    transaction_id: str
    amount: float
    merchant: str
    location: str
    time_of_day: int  # 0-23 hours
    transaction_type: Optional[str] = "payment"
    payment_method: Optional[str] = "card"
    merchant_category: Optional[str] = "general"

class FraudResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    blockchain_tx_id: Optional[str] = None
    fraud_factors: Optional[Dict[str, Any]] = {}
    confidence_score: Optional[float] = None
    risk_level: Optional[str] = "low"
    recommended_action: Optional[str] = "approve"

# ============================================================================
# ENHANCED COMPLIANCE MODELS
# ============================================================================

class ComplianceCheckRequest(BaseModel):
    transaction_amount: float
    kyc_completed: bool
    source_of_funds: str
    beneficial_owner_disclosed: Optional[bool] = True
    is_pep: Optional[bool] = False
    transaction_purpose: Optional[str] = "general"
    country_of_origin: Optional[str] = "IN"

class ComplianceResponse(BaseModel):
    user_id: str
    compliant: bool
    violations: List[str]
    blockchain_tx_id: Optional[str] = None
    risk_score: Optional[float] = None
    compliance_confidence: Optional[float] = None
    recommended_actions: Optional[List[str]] = []
    review_required: Optional[bool] = False

# ============================================================================
# ENHANCED CHAT MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    response: str
    confidence: float
    intent: Optional[str] = None
    suggested_actions: Optional[List[str]] = []
    entities: Optional[Dict[str, Any]] = {}
    follow_up_questions: Optional[List[str]] = []

# ============================================================================
# ENHANCED MARKET DATA MODELS
# ============================================================================

class MarketData(BaseModel):
    symbol: str
    current_price: float
    ma_7: float
    ma_20: Optional[float] = None
    ma_50: Optional[float] = None
    rsi: float
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    volume: int
    avg_volume: Optional[int] = None
    recommendation: str  # BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
    confidence: Optional[float] = None
    ai_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    news_sentiment: Optional[str] = None
    technical_signals: Optional[Dict[str, Any]] = {}
    fundamental_metrics: Optional[Dict[str, Any]] = {}
    timestamp: datetime

class StockInsights(BaseModel):
    symbol: str
    ai_analysis: str
    technical_analysis: Dict[str, Any]
    fundamental_analysis: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    price_prediction: Dict[str, float]
    risk_assessment: Dict[str, Any]
    recommendation_reasoning: str
    confidence_factors: List[str]
    timestamp: datetime

# ============================================================================
# BLOCKCHAIN MODELS
# ============================================================================

class BlockchainTransaction(BaseModel):
    transaction_id: str
    block_number: Optional[int] = None
    transaction_hash: Optional[str] = None
    from_address: Optional[str] = None
    to_address: Optional[str] = None
    gas_used: Optional[int] = None
    gas_price: Optional[int] = None
    timestamp: datetime
    transaction_type: str
    data: Dict[str, Any]
    confirmations: Optional[int] = 0

class UserProfile(BaseModel):
    user_id: str
    email: str
    full_name: str
    profile_hash: str
    blockchain_address: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    verification_status: Optional[str] = "pending"
    kyc_level: Optional[int] = 0

# ============================================================================
# ANALYTICS & REPORTING MODELS
# ============================================================================

class UserAnalytics(BaseModel):
    user_id: str
    total_transactions: int
    total_loan_applications: int
    successful_loans: int
    fraud_incidents: int
    compliance_violations: int
    risk_score: float
    activity_score: float
    last_activity: datetime
    account_age_days: int

class SystemMetrics(BaseModel):
    total_users: int
    active_users_24h: int
    total_transactions: int
    successful_loans: int
    fraud_detection_rate: float
    compliance_rate: float
    system_uptime: float
    api_response_time: float
    blockchain_sync_status: str
    model_accuracy_scores: Dict[str, float]

# ============================================================================
# ERROR MODELS
# ============================================================================

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime
    path: Optional[str] = None
    error_code: Optional[str] = None
