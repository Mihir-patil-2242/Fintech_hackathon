"""
Minimal FinTech Platform Backend - Simplified for Development
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Try to import required packages, with fallbacks
try:
    from fastapi import FastAPI, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("FastAPI not available, using simple HTTP server")
    FASTAPI_AVAILABLE = False

# If FastAPI isn't available, create a simple HTTP server alternative
if not FASTAPI_AVAILABLE:
    import http.server
    import socketserver
    import urllib.parse
    import threading
    
    class SimpleAPIHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {"message": "FinTech Platform API - Minimal Mode", "status": "running"}
                self.wfile.write(json.dumps(response).encode())
            elif self.path == '/docs':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                html = """
                <!DOCTYPE html>
                <html>
                <head><title>FinTech API Docs</title></head>
                <body>
                    <h1>FinTech Platform API Documentation</h1>
                    <h2>Available Endpoints:</h2>
                    <ul>
                        <li>GET / - API status</li>
                        <li>GET /health - Health check</li>
                        <li>GET /market/data - Market data</li>
                        <li>POST /auth/login - Authentication</li>
                        <li>POST /loans/apply - Loan application</li>
                        <li>POST /investments/create - Investment creation</li>
                    </ul>
                </body>
                </html>
                """
                self.wfile.write(html.encode())
            elif self.path == '/health':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {
                    "status": "healthy",
                    "mode": "minimal",
                    "timestamp": datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(404)
                self.end_headers()
        
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()

# Configuration with fallbacks
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fintech.db")
JWT_SECRET = os.getenv("JWT_SECRET", "minimal-dev-secret")

# Pydantic Models (with fallbacks)
if FASTAPI_AVAILABLE:
    class LoginRequest(BaseModel):
        wallet_address: str
        signature: str
        message: str

    class LoanRequest(BaseModel):
        amount: float
        duration: int
        purpose: str
        income: float
        employment: str

    class InvestmentRequest(BaseModel):
        amount: float
        strategy: str = "balanced"

    class ChatMessage(BaseModel):
        message: str
        context: Optional[Dict] = None

# Mock data for development
MOCK_MARKET_DATA = {
    "crypto": {
        "BTC": {"price": 45000.12, "change_24h": 2.5, "volume": 25000000},
        "ETH": {"price": 3200.50, "change_24h": -1.2, "volume": 15000000},
        "ADA": {"price": 0.52, "change_24h": 1.8, "volume": 8000000}
    },
    "stocks": {
        "AAPL": {"price": 175.80, "change": 1.8, "volume": 50000000},
        "GOOGL": {"price": 140.20, "change": -0.5, "volume": 30000000},
        "TSLA": {"price": 248.50, "change": 3.2, "volume": 45000000}
    }
}

MOCK_DEBT_CYCLE = {
    "indicators": {
        "total_debt": 1000000,
        "total_collateral": 1500000,
        "health_score": 75,
        "cycle_phase": "expansion",
        "default_rate": 2.5,
        "credit_growth_rate": 5.2
    }
}

# Simple authentication store (in-memory for development)
active_tokens = {}

def create_simple_token(wallet_address: str) -> str:
    """Create a simple token for development"""
    import hashlib
    token = hashlib.md5(f"{wallet_address}{time.time()}".encode()).hexdigest()
    expire_time = time.time() + (24 * 3600)  # 24 hours
    active_tokens[token] = {"wallet": wallet_address, "expires": expire_time}
    return token

def verify_simple_token(auth_header: str) -> str:
    """Verify a simple token"""
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = auth_header.replace("Bearer ", "")
    token_data = active_tokens.get(token)
    
    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    if token_data["expires"] < time.time():
        del active_tokens[token]
        raise HTTPException(status_code=401, detail="Token expired")
    
    return token_data["wallet"]

if FASTAPI_AVAILABLE:
    # Initialize FastAPI app
    app = FastAPI(
        title="FinTech Platform API - Minimal",
        description="Simplified FinTech platform for development",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {
            "message": "FinTech Platform API - Minimal Mode",
            "status": "running",
            "docs": "Visit /docs for API documentation",
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "mode": "minimal",
            "blockchain_connected": False,  # Simplified for development
            "database": True,
            "ai_enabled": False,  # Simplified for development
            "redis_connected": False,  # Simplified for development
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/auth/nonce")
    async def get_nonce(wallet_address: str):
        import hashlib
        nonce = hashlib.sha256(f"{wallet_address}{time.time()}".encode()).hexdigest()
        message = f"Sign this message to authenticate with FinTech Platform.\n\nNonce: {nonce}"
        return {"message": message}

    @app.post("/auth/login")
    async def login(request: LoginRequest):
        try:
            # Simplified auth for development - just create a token
            token = create_simple_token(request.wallet_address)
            
            return {
                "access_token": token,
                "token_type": "bearer",
                "credit_score": 720  # Mock credit score
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Login failed: {str(e)}")

    @app.post("/loans/apply")
    async def apply_for_loan(request: LoanRequest):
        # Mock loan processing
        collateral_ratio = 1.5 if request.amount < 10000 else 2.0
        interest_rate = 8.5 if request.income > 50000 else 12.0
        
        return {
            "loan_application": {
                "amount": request.amount,
                "collateral_required": request.amount * collateral_ratio,
                "interest_rate": interest_rate,
                "status": "approved",
                "loan_id": f"LOAN_{int(time.time())}"
            }
        }

    @app.post("/investments/create")
    async def create_investment(request: InvestmentRequest):
        apy_map = {"conservative": 6.0, "balanced": 9.0, "aggressive": 13.5}
        expected_apy = apy_map.get(request.strategy, 9.0)
        
        return {
            "investment_plan": {
                "amount": request.amount,
                "strategy": request.strategy,
                "expected_apy": expected_apy,
                "investment_id": f"INV_{int(time.time())}"
            }
        }

    @app.post("/chat")
    async def chat_with_ai(message: ChatMessage):
        # Mock AI responses
        responses = {
            "credit": "Your credit score is looking good! Consider diversifying your portfolio for better returns.",
            "investment": "Based on current market conditions, a balanced portfolio with 60% stocks and 40% bonds could be optimal.",
            "loan": "With your current credit profile, you're eligible for competitive rates. Consider shorter terms for lower total interest.",
            "market": "Current market trends show steady growth in tech stocks and increasing adoption of DeFi protocols."
        }
        
        # Simple keyword matching for demo
        response_text = "I'm here to help with your financial decisions! Ask me about loans, investments, or market analysis."
        for keyword, resp in responses.items():
            if keyword in message.message.lower():
                response_text = resp
                break
        
        return {"response": response_text}

    @app.get("/market/data")
    async def get_market_data():
        return MOCK_MARKET_DATA

    @app.get("/market/debt-cycle")
    async def get_debt_cycle_analysis():
        return MOCK_DEBT_CYCLE

def start_minimal_server():
    """Start server with minimal dependencies"""
    if FASTAPI_AVAILABLE:
        print("🚀 Starting FinTech Platform API (FastAPI Mode)...")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔍 Alternative docs: http://localhost:8000/redoc")
        print("🌐 API Root: http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    else:
        print("🚀 Starting FinTech Platform API (Simple HTTP Mode)...")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🌐 API Root: http://localhost:8000")
        
        with socketserver.TCPServer(("", 8000), SimpleAPIHandler) as httpd:
            httpd.serve_forever()

if __name__ == "__main__":
    start_minimal_server()
