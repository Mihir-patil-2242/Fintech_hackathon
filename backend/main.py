from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import uuid
from datetime import datetime
from typing import List

# Import models
from models.loan_request import (
    LoanRequest, LoanResponse, FraudCheckRequest, FraudResponse,
    ComplianceCheckRequest, ComplianceResponse, ChatRequest, ChatResponse,
    MarketData
)

# Import services
from services.credit_scoring import CreditScoringService
from services.fraud_detection import FraudDetectionService
from services.investment_advice import InvestmentAdviceService
from services.compliance_check import ComplianceService
from services.chatbot import EnhancedChatbotService
from services.blockchain import EnhancedBlockchainService

# Initialize FastAPI app
app = FastAPI(
    title="FinTech AI Platform",
    description="Blockchain + AI-powered FinTech platform for hackathon demo",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
credit_service = CreditScoringService()
fraud_service = FraudDetectionService()
investment_service = InvestmentAdviceService()
compliance_service = ComplianceService()
chatbot_service = EnhancedChatbotService()
blockchain_service = EnhancedBlockchainService()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "FinTech AI Platform API",
        "version": "1.0.0",
        "endpoints": {
            "loan_request": "/loan-request",
            "fraud_check": "/check-fraud", 
            "compliance_check": "/compliance-check",
            "market_data": "/market-data",
            "chatbot": "/chatbot",
            "transactions": "/transactions",
            "blockchain_stats": "/blockchain/stats"
        }
    }

@app.post("/loan-request", response_model=LoanResponse)
async def process_loan_request(request: LoanRequest):
    """Process loan application with AI credit scoring and blockchain logging"""
    try:
        # Generate unique loan ID
        loan_id = f"LOAN_{uuid.uuid4().hex[:8].upper()}"
        
        # Calculate credit score using AI
        loan_data = request.dict()
        credit_score = credit_service.calculate_credit_score(loan_data)
        
        # Check loan approval
        approved, reason = credit_service.approve_loan(credit_score, request.loan_amount)
        
        # Check for fraud
        fraud_check_data = {
            'amount': request.loan_amount,
            'time_of_day': datetime.now().hour,
            'location': 'loan_application',
            'merchant': 'bank_loan_dept'
        }
        fraud_prob, is_fraud = fraud_service.detect_fraud(fraud_check_data)
        
        # Override approval if fraud detected
        if is_fraud:
            approved = False
            reason = f"Fraud detected (probability: {fraud_prob:.2f})"
        
        # Create response
        response = LoanResponse(
            loan_id=loan_id,
            approved=approved,
            credit_score=credit_score,
            reason=reason
        )
        
        # Log to blockchain
        blockchain_data = {
            'user_id': request.user_id,
            'loan_id': loan_id,
            'approved': approved,
            'credit_score': credit_score,
            'loan_amount': request.loan_amount,
            'reason': reason,
            'fraud_probability': fraud_prob
        }
        
        tx_id = blockchain_service.add_loan_decision(blockchain_data)
        response.blockchain_tx_id = tx_id
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing loan request: {str(e)}")

@app.post("/check-fraud", response_model=FraudResponse)
async def check_fraud(request: FraudCheckRequest):
    """Check transaction for fraud using AI detection"""
    try:
        # Prepare fraud detection data
        fraud_data = {
            'amount': request.amount,
            'time_of_day': request.time_of_day,
            'location': request.location,
            'merchant': request.merchant
        }
        
        # Run fraud detection
        fraud_probability, is_fraud = fraud_service.detect_fraud(fraud_data)
        
        # Create response
        response = FraudResponse(
            transaction_id=request.transaction_id,
            fraud_probability=fraud_probability,
            is_fraud=is_fraud
        )
        
        # Log to blockchain if fraud detected
        if is_fraud:
            blockchain_data = {
                'transaction_id': request.transaction_id,
                'user_id': request.user_id,
                'fraud_probability': fraud_probability,
                'is_fraud': is_fraud,
                'amount': request.amount
            }
            
            tx_id = blockchain_service.add_fraud_alert(blockchain_data)
            response.blockchain_tx_id = tx_id
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking fraud: {str(e)}")

@app.post("/compliance-check", response_model=ComplianceResponse)
async def check_compliance(request: ComplianceCheckRequest):
    """Check AML/KYC compliance"""
    try:
        # Run compliance check
        is_compliant, violations = compliance_service.check_compliance(request.dict())
        
        # Create response
        response = ComplianceResponse(
            user_id=request.user_id,
            compliant=is_compliant,
            violations=violations
        )
        
        # Log to blockchain if violations found
        if not is_compliant:
            compliance_report = compliance_service.generate_compliance_report(violations)
            
            blockchain_data = {
                'user_id': request.user_id,
                'compliant': is_compliant,
                'violations': violations,
                'risk_level': compliance_report.get('risk_level'),
                'transaction_amount': request.transaction_amount
            }
            
            tx_id = blockchain_service.add_compliance_alert(blockchain_data)
            response.blockchain_tx_id = tx_id
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking compliance: {str(e)}")

@app.get("/market-data")
async def get_market_data(symbols: str = "AAPL,GOOGL,MSFT,TSLA,RELIANCE.NS"):
    """Get live market data and investment advice"""
    try:
        symbol_list = [s.strip() for s in symbols.split(',')]
        market_data = investment_service.get_portfolio_advice(symbol_list)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "data": market_data,
            "count": len(market_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")

@app.get("/market-data/popular")
async def get_popular_stocks():
    """Get data for popular stocks"""
    try:
        market_data = investment_service.get_popular_stocks()
        return {
            "timestamp": datetime.now().isoformat(),
            "data": market_data,
            "count": len(market_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching popular stocks: {str(e)}")

@app.post("/chatbot", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """Chat with AI financial assistant"""
    try:
        # Process message
        response_text, confidence, intent = chatbot_service.process_message(
            request.user_id, request.message
        )
        
        # Create chat log for blockchain
        chat_data = {
            'user_id': request.user_id,
            'message': request.message,
            'response': response_text,
            'intent': intent,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log chat hash to blockchain
        blockchain_service.add_chat_log(chat_data)
        
        return ChatResponse(
            response=response_text,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/transactions")
async def get_transactions(limit: int = 20):
    """Get recent blockchain transactions"""
    try:
        transactions = blockchain_service.get_recent_transactions(limit)
        return {
            "transactions": transactions,
            "count": len(transactions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transactions: {str(e)}")

@app.get("/transactions/user/{user_id}")
async def get_user_transactions(user_id: str):
    """Get transactions for specific user"""
    try:
        transactions = blockchain_service.get_user_transactions(user_id)
        return {
            "user_id": user_id,
            "transactions": transactions,
            "count": len(transactions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user transactions: {str(e)}")

@app.get("/blockchain/stats")
async def get_blockchain_stats():
    """Get blockchain statistics"""
    try:
        stats = blockchain_service.get_blockchain_stats()
        return {
            "blockchain_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching blockchain stats: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "credit_scoring": "active",
            "fraud_detection": "active", 
            "investment_advice": "active",
            "compliance_check": "active",
            "chatbot": "active",
            "blockchain": "active"
        }
    }

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    print("🚀 Starting FinTech AI Platform...")
    print("📊 Services initialized:")
    print("   ✅ Credit Scoring AI")
    print("   ✅ Fraud Detection AI") 
    print("   ✅ Investment Advice AI")
    print("   ✅ Compliance Check AI")
    print("   ✅ Chatbot AI")
    print("   ✅ Blockchain Service")
    print("\n🌐 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
