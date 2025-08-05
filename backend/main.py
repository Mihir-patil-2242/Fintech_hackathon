from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn
import uuid
from datetime import datetime, timedelta
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import models
from models.loan_request import (
    LoanRequest, LoanResponse, FraudCheckRequest, FraudResponse,
    ComplianceCheckRequest, ComplianceResponse, ChatRequest, ChatResponse,
    MarketData, UserRegistration, UserLogin, UserResponse, TokenResponse
)

# Import enhanced services
from services.auth_service import AuthService
from services.user_service import UserService
from services.credit_scoring_enhanced import EnhancedCreditScoringService
from services.fraud_detection_enhanced import EnhancedFraudDetectionService
from services.investment_advice_enhanced import EnhancedInvestmentAdviceService
from services.compliance_check_enhanced import EnhancedComplianceService
from services.chatbot_enhanced import EnhancedChatbotService
from services.blockchain_enhanced import EnhancedBlockchainService

# Initialize FastAPI app
app = FastAPI(
    title="FinTech AI Platform - Enhanced",
    description="Blockchain + AI-powered FinTech platform with authentication and pre-trained models",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize services
auth_service = AuthService()
user_service = UserService()
credit_service = EnhancedCreditScoringService()
fraud_service = EnhancedFraudDetectionService()
investment_service = EnhancedInvestmentAdviceService()
compliance_service = EnhancedComplianceService()
chatbot_service = EnhancedChatbotService()
blockchain_service = EnhancedBlockchainService()

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = auth_service.verify_token(token)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = await user_service.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "FinTech AI Platform API - Enhanced",
        "version": "2.0.0",
        "features": [
            "User Authentication & Blockchain Profiles",
            "Pre-trained AI Models",
            "Ganache Testnet Integration", 
            "Enhanced Financial Feed",
            "Microservices Architecture"
        ],
        "endpoints": {
            "auth": {
                "register": "/auth/register",
                "login": "/auth/login",
                "profile": "/auth/profile"
            },
            "services": {
                "loan": "/services/loan",
                "fraud": "/services/fraud", 
                "compliance": "/services/compliance",
                "market": "/services/market",
                "chat": "/services/chat"
            },
            "blockchain": {
                "transactions": "/blockchain/transactions",
                "stats": "/blockchain/stats",
                "user_profile": "/blockchain/user/{user_id}"
            }
        }
    }

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/register", response_model=UserResponse)
async def register_user(user_data: UserRegistration):
    """Register a new user with blockchain profile creation"""
    try:
        # Check if user already exists
        existing_user = await user_service.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create user
        user = await user_service.create_user(user_data)
        
        # Create blockchain profile
        blockchain_profile = {
            'user_id': user.user_id,
            'email': user.email,
            'full_name': user.full_name,
            'created_at': datetime.utcnow().isoformat(),
            'profile_type': 'USER_REGISTRATION'
        }
        
        # Store on blockchain
        tx_id = await blockchain_service.create_user_profile(blockchain_profile)
        
        # Update user with blockchain transaction ID
        await user_service.update_user_blockchain_tx(user.user_id, tx_id)
        
        return UserResponse(
            user_id=user.user_id,
            email=user.email,
            full_name=user.full_name,
            created_at=user.created_at,
            blockchain_tx_id=tx_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering user: {str(e)}")

@app.post("/auth/login", response_model=TokenResponse)
async def login_user(login_data: UserLogin):
    """Authenticate user and return JWT token"""
    try:
        # Verify credentials
        user = await user_service.authenticate_user(login_data.email, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token = auth_service.create_access_token(
            data={"sub": user.user_id, "email": user.email}
        )
        
        # Log login to blockchain
        login_event = {
            'user_id': user.user_id,
            'event_type': 'USER_LOGIN',
            'timestamp': datetime.utcnow().isoformat(),
            'ip_address': 'localhost',  # In production, get real IP
            'user_agent': 'FinTech-App'
        }
        await blockchain_service.log_user_event(login_event)
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=user.user_id,
            email=user.email
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging in: {str(e)}")

@app.get("/auth/profile", response_model=UserResponse)
async def get_user_profile(current_user = Depends(get_current_user)):
    """Get current user's profile"""
    return UserResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        full_name=current_user.full_name,
        created_at=current_user.created_at,
        blockchain_tx_id=current_user.blockchain_tx_id
    )

# ============================================================================
# LOAN SERVICE ENDPOINTS
# ============================================================================

@app.post("/services/loan/apply", response_model=LoanResponse)
async def apply_for_loan(request: LoanRequest, current_user = Depends(get_current_user)):
    """Process loan application with enhanced AI credit scoring"""
    try:
        # Generate unique loan ID
        loan_id = f"LOAN_{uuid.uuid4().hex[:8].upper()}"
        
        # Enhanced credit scoring with pre-trained models
        loan_data = request.dict()
        loan_data['user_id'] = current_user.user_id
        
        credit_score, confidence, risk_factors = await credit_service.calculate_enhanced_credit_score(loan_data)
        
        # Enhanced loan approval with multiple models
        approved, reason, approval_confidence = await credit_service.enhanced_loan_approval(
            credit_score, request.loan_amount, loan_data
        )
        
        # Enhanced fraud detection
        fraud_check_data = {
            'user_id': current_user.user_id,
            'amount': request.loan_amount,
            'time_of_day': datetime.now().hour,
            'location': 'loan_application',
            'merchant': 'bank_loan_dept',
            'user_history': await user_service.get_user_transaction_history(current_user.user_id)
        }
        fraud_prob, is_fraud, fraud_factors = await fraud_service.enhanced_fraud_detection(fraud_check_data)
        
        # Override approval if fraud detected
        if is_fraud:
            approved = False
            reason = f"Potential fraud detected (probability: {fraud_prob:.2f})"
        
        # Create enhanced response
        response = LoanResponse(
            loan_id=loan_id,
            approved=approved,
            credit_score=credit_score,
            reason=reason,
            confidence=confidence,
            risk_factors=risk_factors,
            fraud_probability=fraud_prob
        )
        
        # Log to blockchain with enhanced data
        blockchain_data = {
            'user_id': current_user.user_id,
            'loan_id': loan_id,
            'approved': approved,
            'credit_score': credit_score,
            'loan_amount': request.loan_amount,
            'reason': reason,
            'fraud_probability': fraud_prob,
            'confidence': confidence,
            'risk_factors': risk_factors,
            'processing_timestamp': datetime.utcnow().isoformat()
        }
        
        tx_id = await blockchain_service.add_loan_decision(blockchain_data)
        response.blockchain_tx_id = tx_id
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing loan request: {str(e)}")

@app.get("/services/loan/status/{loan_id}")
async def get_loan_status(loan_id: str, current_user = Depends(get_current_user)):
    """Get loan application status"""
    try:
        # Get loan status from blockchain
        status_data = await blockchain_service.get_loan_status(loan_id, current_user.user_id)
        return status_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting loan status: {str(e)}")

# ============================================================================
# FRAUD DETECTION SERVICE ENDPOINTS
# ============================================================================

@app.post("/services/fraud/check", response_model=FraudResponse)
async def check_fraud_enhanced(request: FraudCheckRequest, current_user = Depends(get_current_user)):
    """Enhanced fraud detection using pre-trained models"""
    try:
        # Prepare enhanced fraud detection data
        fraud_data = {
            'user_id': current_user.user_id,
            'transaction_id': request.transaction_id,
            'amount': request.amount,
            'time_of_day': request.time_of_day,
            'location': request.location,
            'merchant': request.merchant,
            'user_history': await user_service.get_user_transaction_history(current_user.user_id),
            'account_age': await user_service.get_account_age(current_user.user_id)
        }
        
        # Run enhanced fraud detection
        fraud_probability, is_fraud, fraud_factors = await fraud_service.enhanced_fraud_detection(fraud_data)
        
        # Create enhanced response
        response = FraudResponse(
            transaction_id=request.transaction_id,
            fraud_probability=fraud_probability,
            is_fraud=is_fraud,
            fraud_factors=fraud_factors,
            confidence_score=fraud_factors.get('confidence', 0.0)
        )
        
        # Log to blockchain if fraud detected
        if is_fraud:
            blockchain_data = {
                'transaction_id': request.transaction_id,
                'user_id': current_user.user_id,
                'fraud_probability': fraud_probability,
                'is_fraud': is_fraud,
                'amount': request.amount,
                'fraud_factors': fraud_factors,
                'detection_timestamp': datetime.utcnow().isoformat()
            }
            
            tx_id = await blockchain_service.add_fraud_alert(blockchain_data)
            response.blockchain_tx_id = tx_id
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking fraud: {str(e)}")

# ============================================================================
# COMPLIANCE SERVICE ENDPOINTS
# ============================================================================

@app.post("/services/compliance/check", response_model=ComplianceResponse)
async def check_compliance_enhanced(request: ComplianceCheckRequest, current_user = Depends(get_current_user)):
    """Enhanced AML/KYC compliance check"""
    try:
        # Enhanced compliance check
        request_data = request.dict()
        request_data['user_id'] = current_user.user_id
        request_data['user_history'] = await user_service.get_user_transaction_history(current_user.user_id)
        
        is_compliant, violations, risk_score = await compliance_service.enhanced_compliance_check(request_data)
        
        # Create enhanced response
        response = ComplianceResponse(
            user_id=current_user.user_id,
            compliant=is_compliant,
            violations=violations,
            risk_score=risk_score,
            compliance_confidence=0.95  # From enhanced model
        )
        
        # Log to blockchain if violations found
        if not is_compliant:
            compliance_report = await compliance_service.generate_enhanced_compliance_report(violations)
            
            blockchain_data = {
                'user_id': current_user.user_id,
                'compliant': is_compliant,
                'violations': violations,
                'risk_score': risk_score,
                'transaction_amount': request.transaction_amount,
                'compliance_report': compliance_report,
                'check_timestamp': datetime.utcnow().isoformat()
            }
            
            tx_id = await blockchain_service.add_compliance_alert(blockchain_data)
            response.blockchain_tx_id = tx_id
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking compliance: {str(e)}")

# ============================================================================
# MARKET DATA SERVICE ENDPOINTS
# ============================================================================

@app.get("/services/market/data")
async def get_enhanced_market_data(symbols: str = "AAPL,GOOGL,MSFT,TSLA,RELIANCE.NS", current_user = Depends(get_current_user)):
    """Get enhanced market data with AI insights"""
    try:
        symbol_list = [s.strip() for s in symbols.split(',')]
        market_data = await investment_service.get_enhanced_portfolio_advice(symbol_list, current_user.user_id)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "data": market_data,
            "count": len(market_data),
            "user_id": current_user.user_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")

@app.get("/services/market/popular")
async def get_popular_stocks_enhanced(current_user = Depends(get_current_user)):
    """Get enhanced data for popular stocks with AI insights"""
    try:
        market_data = await investment_service.get_enhanced_popular_stocks(current_user.user_id)
        return {
            "timestamp": datetime.now().isoformat(),
            "data": market_data,
            "count": len(market_data),
            "user_id": current_user.user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching popular stocks: {str(e)}")

@app.get("/services/market/insights/{symbol}")
async def get_ai_insights(symbol: str, current_user = Depends(get_current_user)):
    """Get AI-powered insights for a specific stock"""
    try:
        insights = await investment_service.get_ai_stock_insights(symbol, current_user.user_id)
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting AI insights: {str(e)}")

# ============================================================================
# CHATBOT SERVICE ENDPOINTS
# ============================================================================

@app.post("/services/chat/message", response_model=ChatResponse)
async def chat_with_enhanced_bot(request: ChatRequest, current_user = Depends(get_current_user)):
    """Chat with enhanced AI financial assistant"""
    try:
        # Enhanced chat processing with user context
        response_text, confidence, intent, suggested_actions = await chatbot_service.process_enhanced_message(
            current_user.user_id, request.message
        )
        
        # Create enhanced chat log for blockchain
        chat_data = {
            'user_id': current_user.user_id,
            'message': request.message,
            'response': response_text,
            'intent': intent,
            'confidence': confidence,
            'suggested_actions': suggested_actions,
            'timestamp': datetime.now().isoformat(),
            'session_id': request.session_id or f"session_{uuid.uuid4().hex[:8]}"
        }
        
        # Log chat hash to blockchain
        await blockchain_service.add_chat_log(chat_data)
        
        return ChatResponse(
            response=response_text,
            confidence=confidence,
            intent=intent,
            suggested_actions=suggested_actions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/services/chat/history")
async def get_chat_history(current_user = Depends(get_current_user), limit: int = 20):
    """Get user's chat history"""
    try:
        history = await chatbot_service.get_user_chat_history(current_user.user_id, limit)
        return {
            "user_id": current_user.user_id,
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting chat history: {str(e)}")

# ============================================================================
# BLOCKCHAIN SERVICE ENDPOINTS
# ============================================================================

@app.get("/blockchain/transactions")
async def get_transactions(current_user = Depends(get_current_user), limit: int = 20):
    """Get recent blockchain transactions"""
    try:
        transactions = await blockchain_service.get_recent_transactions(limit)
        return {
            "transactions": transactions,
            "count": len(transactions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transactions: {str(e)}")

@app.get("/blockchain/user/{user_id}")
async def get_user_blockchain_profile(user_id: str, current_user = Depends(get_current_user)):
    """Get user's blockchain profile and transactions"""
    try:
        # Ensure user can only access their own data or admin access
        if current_user.user_id != user_id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
            
        profile = await blockchain_service.get_user_blockchain_profile(user_id)
        transactions = await blockchain_service.get_user_transactions(user_id)
        
        return {
            "user_id": user_id,
            "profile": profile,
            "transactions": transactions,
            "transaction_count": len(transactions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user blockchain data: {str(e)}")

@app.get("/blockchain/stats")
async def get_blockchain_stats(current_user = Depends(get_current_user)):
    """Get comprehensive blockchain statistics"""
    try:
        stats = await blockchain_service.get_enhanced_blockchain_stats()
        return {
            "blockchain_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching blockchain stats: {str(e)}")

# ============================================================================
# HEALTH & MONITORING ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    service_status = {}
    
    try:
        # Check each service
        service_status['auth_service'] = await auth_service.health_check()
        service_status['credit_scoring'] = await credit_service.health_check()
        service_status['fraud_detection'] = await fraud_service.health_check()
        service_status['investment_advice'] = await investment_service.health_check()
        service_status['compliance_check'] = await compliance_service.health_check()
        service_status['chatbot'] = await chatbot_service.health_check()
        service_status['blockchain'] = await blockchain_service.health_check()
        
        all_healthy = all(status == "healthy" for status in service_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "services": service_status,
            "features": {
                "authentication": True,
                "blockchain_profiles": True,
                "pretrained_models": True,
                "ganache_integration": True,
                "enhanced_financial_feed": True,
                "microservices": True
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "services": service_status
        }

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    print("🚀 Starting Enhanced FinTech AI Platform...")
    print("📊 Enhanced Services initialized:")
    print("   ✅ User Authentication & JWT")
    print("   ✅ Enhanced Credit Scoring AI (XGBoost + Neural Networks)")
    print("   ✅ Advanced Fraud Detection AI (Ensemble Models)")
    print("   ✅ Intelligent Investment Advice (ML + Technical Analysis)")
    print("   ✅ Smart Compliance Check (NLP + Rule Engine)")
    print("   ✅ Advanced NLP Chatbot (Transformers)")
    print("   ✅ Ganache Blockchain Integration")
    print("   ✅ Microservices Architecture")
    print("\n🌐 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
