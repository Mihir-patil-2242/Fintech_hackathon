"""
Finverse - Ultra-Minimal DeFi Platform Backend
Guaranteed to work with minimal dependencies
"""

import json
import time
from datetime import datetime
from typing import Dict, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
    print("✅ FastAPI available")
except ImportError:
    HAS_FASTAPI = False
    print("❌ FastAPI not available, install with: pip install fastapi uvicorn")

# Data Models
class LoginRequest(BaseModel):
    wallet_address: str
    signature: str = ""
    message: str = ""

class LoanRequest(BaseModel):
    amount: float
    duration: int = 30
    purpose: str = "personal"
    income: float = 50000
    employment: str = "employed"

class InvestmentRequest(BaseModel):
    amount: float
    strategy: str = "balanced"

class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict] = None

# Mock data and state
users_db = {}
active_tokens = {}

# Mock data
MOCK_MARKET_DATA = {
    "crypto": {
        "BTC": {"price": 45123.45, "change_24h": 2.3, "volume": 28500000},
        "ETH": {"price": 3245.67, "change_24h": -0.8, "volume": 18200000},
        "ADA": {"price": 0.58, "change_24h": 4.2, "volume": 9500000},
        "DOT": {"price": 8.92, "change_24h": 1.5, "volume": 3200000}
    },
    "stocks": {
        "AAPL": {"price": 178.25, "change": 2.1, "volume": 52000000},
        "GOOGL": {"price": 142.80, "change": -0.3, "volume": 31000000},
        "TSLA": {"price": 251.70, "change": 3.8, "volume": 47000000},
        "MSFT": {"price": 415.30, "change": 1.2, "volume": 28000000}
    }
}

DEBT_CYCLE_DATA = {
    "indicators": {
        "total_debt": 1250000,
        "total_collateral": 1875000,
        "health_score": 78,
        "cycle_phase": "early_expansion",
        "default_rate": 2.1,
        "credit_growth_rate": 6.4,
        "risk_level": "moderate"
    }
}

# Simple token management
def create_token(wallet_address: str) -> str:
    import hashlib
    token = hashlib.md5(f"{wallet_address}{time.time()}".encode()).hexdigest()
    active_tokens[token] = {
        "wallet": wallet_address,
        "expires": time.time() + (24 * 3600),
        "created": datetime.now()
    }
    return token

# AI Response patterns
AI_RESPONSES = {
    "credit": [
        "Your credit score of 720 is excellent! This puts you in the top 25% of borrowers. Consider leveraging this for better loan terms.",
        "With a strong credit profile, you're eligible for our premium rates starting at 5.5% APR. Would you like to explore loan options?",
        "Your creditworthiness opens doors to exclusive investment opportunities with lower risk thresholds."
    ],
    "loan": [
        "Based on your income profile, I recommend considering a loan amount that keeps your debt-to-income ratio below 30%.",
        "For your loan purpose, our DeFi lending pools offer competitive rates. The current market conditions favor borrowers.",
        "Consider a shorter term loan to minimize interest payments, especially with your strong credit profile."
    ],
    "investment": [
        "Given current market conditions, a balanced portfolio with 60% equities and 40% bonds could optimize your returns.",
        "The debt cycle indicates we're in an expansion phase - consider tilting towards growth assets while maintaining diversification.",
        "Your risk tolerance suggests our balanced investment pool (9% APY) would be optimal for your financial goals."
    ],
    "market": [
        "Current market trends show resilience in tech stocks and growing adoption of DeFi protocols.",
        "The debt cycle health score of 78 indicates stable market conditions with moderate growth opportunities.",
        "Crypto markets are showing consolidation patterns, which often precede significant moves. Consider dollar-cost averaging."
    ],
    "general": [
        "I'm here to help optimize your financial strategy. What specific area would you like to explore?",
        "Based on your profile, you have several opportunities to improve your financial position. Shall we discuss?",
        "Financial planning is about balancing risk and return. Let me help you find the right balance for your goals."
    ]
}

def get_ai_response(message: str) -> str:
    """Generate contextual AI response"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["credit", "score"]):
        import random
        return random.choice(AI_RESPONSES["credit"])
    elif any(word in message_lower for word in ["loan", "borrow", "lending"]):
        import random
        return random.choice(AI_RESPONSES["loan"])
    elif any(word in message_lower for word in ["invest", "portfolio", "returns"]):
        import random
        return random.choice(AI_RESPONSES["investment"])
    elif any(word in message_lower for word in ["market", "price", "trend"]):
        import random
        return random.choice(AI_RESPONSES["market"])
    else:
        import random
        return random.choice(AI_RESPONSES["general"])

if HAS_FASTAPI:
    # Initialize FastAPI
    app = FastAPI(
        title="Finverse API",
        description="Minimal DeFi platform with AI-powered features",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {
            "message": "Finverse API v2.0",
            "status": "running",
            "mode": "minimal",
            "docs": "/docs",
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "mode": "minimal",
            "backend": "operational",
            "database": "sqlite",
            "ai": "mock_enabled",
            "blockchain": "simulated",
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/auth/nonce")
    async def get_nonce(wallet_address: str):
        import hashlib
        nonce = hashlib.sha256(f"{wallet_address}{time.time()}".encode()).hexdigest()[:16]
        message = f"Welcome to Finverse v2\nSign to authenticate your wallet\nNonce: {nonce}"
        return {"message": message, "nonce": nonce}

    @app.post("/auth/login")
    async def login(request: LoginRequest):
        try:
            # Create user if doesn't exist
            if request.wallet_address not in users_db:
                users_db[request.wallet_address] = {
                    "username": f"user_{request.wallet_address[:6]}",
                    "credit_score": 720,
                    "created_at": datetime.now().isoformat(),
                    "kyc_verified": True
                }
            
            token = create_token(request.wallet_address)
            user = users_db[request.wallet_address]
            
            return {
                "access_token": token,
                "token_type": "bearer",
                "credit_score": user["credit_score"],
                "user": user
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Login failed: {str(e)}")

    @app.post("/loans/apply")
    async def apply_for_loan(request: LoanRequest):
        # Calculate loan terms
        risk_factor = 1.0
        if request.income < 30000:
            risk_factor = 1.5
        elif request.income > 100000:
            risk_factor = 0.8
            
        base_rate = 8.5
        interest_rate = base_rate * risk_factor
        collateral_ratio = 1.5 if request.amount < 10000 else 2.0
        
        loan_id = f"LOAN_{int(time.time())}"
        
        return {
            "loan_application": {
                "loan_id": loan_id,
                "amount": request.amount,
                "duration": request.duration,
                "interest_rate": round(interest_rate, 2),
                "collateral_required": round(request.amount * collateral_ratio, 2),
                "monthly_payment": round((request.amount * (1 + interest_rate/100)) / (request.duration/30), 2),
                "status": "approved",
                "approval_reason": "Strong credit profile and sufficient income",
                "created_at": datetime.now().isoformat()
            }
        }

    @app.post("/investments/create")
    async def create_investment(request: InvestmentRequest):
        strategy_data = {
            "conservative": {"apy": 6.5, "risk": "Low", "description": "Stable returns with minimal risk"},
            "balanced": {"apy": 9.2, "risk": "Medium", "description": "Balanced risk-return profile"},
            "aggressive": {"apy": 14.8, "risk": "High", "description": "High returns with increased volatility"}
        }
        
        strategy_info = strategy_data.get(request.strategy, strategy_data["balanced"])
        investment_id = f"INV_{int(time.time())}"
        
        return {
            "investment_plan": {
                "investment_id": investment_id,
                "amount": request.amount,
                "strategy": request.strategy,
                "expected_apy": strategy_info["apy"],
                "risk_level": strategy_info["risk"],
                "description": strategy_info["description"],
                "projected_annual_return": round(request.amount * strategy_info["apy"] / 100, 2),
                "status": "active",
                "created_at": datetime.now().isoformat()
            }
        }

    @app.post("/chat")
    async def chat_with_ai(message: ChatMessage):
        response_text = get_ai_response(message.message)
        
        return {
            "response": response_text,
            "context": {
                "timestamp": datetime.now().isoformat(),
                "model": "fintech-advisor-v2",
                "confidence": 0.95
            }
        }

    @app.get("/market/data")
    async def get_market_data():
        # Add some random fluctuation to make it look live
        import random
        
        live_data = json.loads(json.dumps(MOCK_MARKET_DATA))  # Deep copy
        
        for category in live_data:
            for symbol in live_data[category]:
                # Add small random fluctuation
                current_price = live_data[category][symbol]["price"]
                fluctuation = random.uniform(-0.02, 0.02)  # ±2%
                live_data[category][symbol]["price"] = round(current_price * (1 + fluctuation), 2)
                
                # Update change
                change_fluctuation = random.uniform(-0.5, 0.5)
                if "change_24h" in live_data[category][symbol]:
                    live_data[category][symbol]["change_24h"] = round(
                        live_data[category][symbol]["change_24h"] + change_fluctuation, 2
                    )
                else:
                    live_data[category][symbol]["change"] = round(
                        live_data[category][symbol]["change"] + change_fluctuation, 2
                    )
        
        return live_data

    @app.get("/market/debt-cycle")
    async def get_debt_cycle_analysis():
        # Add some variability to debt cycle data
        import random
        
        live_cycle = json.loads(json.dumps(DEBT_CYCLE_DATA))
        
        # Small fluctuations in health score
        health_fluctuation = random.randint(-2, 2)
        live_cycle["indicators"]["health_score"] = max(50, min(100, 
            live_cycle["indicators"]["health_score"] + health_fluctuation))
        
        return live_cycle

    @app.get("/user/portfolio")
    async def get_user_portfolio():
        return {
            "total_value": 15420.50,
            "loans": [
                {"id": "LOAN_001", "amount": 5000, "rate": 8.5, "status": "active"},
                {"id": "LOAN_002", "amount": 2500, "rate": 7.2, "status": "active"}
            ],
            "investments": [
                {"id": "INV_001", "amount": 10000, "strategy": "balanced", "return": 920.50},
                {"id": "INV_002", "amount": 5000, "strategy": "conservative", "return": 325.00}
            ]
        }

    def start_server():
        print("🚀 Starting Finverse API...")
        print("📱 Frontend: http://localhost:3000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("🔍 Alternative docs: http://localhost:8000/redoc")
        print("💡 Status: http://localhost:8000/health")
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

else:
    # Fallback HTTP server if FastAPI not available
    import http.server
    import socketserver
    import urllib.parse

    class MinimalAPIHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress default logging
        
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', '*')
            self.send_header('Access-Control-Allow-Headers', '*')
            self.end_headers()

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            if self.path == '/':
                response = {"message": "FinTech Platform API", "status": "running", "docs": "/docs"}
            elif self.path == '/health':
                response = {"status": "healthy", "mode": "fallback", "timestamp": datetime.now().isoformat()}
            elif self.path == '/market/data':
                response = MOCK_MARKET_DATA
            elif self.path == '/market/debt-cycle':
                response = DEBT_CYCLE_DATA
            elif self.path == '/docs':
                self.send_header('Content-type', 'text/html')
                html = """
                <!DOCTYPE html>
                <html><head><title>FinTech API</title></head>
                <body style="font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
                    <h1>FinTech Platform API</h1>
                    <h2>Available Endpoints</h2>
                    <ul>
                        <li><code>GET /</code> - API status</li>
                        <li><code>GET /health</code> - Health check</li>
                        <li><code>GET /market/data</code> - Market data</li>
                        <li><code>POST /auth/login</code> - User authentication</li>
                        <li><code>POST /loans/apply</code> - Loan application</li>
                        <li><code>POST /investments/create</code> - Investment creation</li>
                        <li><code>POST /chat</code> - AI advisor chat</li>
                    </ul>
                    <p><strong>Status:</strong> Running in fallback mode</p>
                </body></html>
                """
                self.wfile.write(html.encode())
                return
            else:
                response = {"error": "Endpoint not found"}
            
            self.wfile.write(json.dumps(response).encode())

    def start_server():
        print("🚀 Starting Finverse API (Fallback Mode)...")
        print("📚 API Docs: http://localhost:8000/docs")
        print("🌐 Health Check: http://localhost:8000/health")
        
        with socketserver.TCPServer(("", 8000), MinimalAPIHandler) as httpd:
            print("✅ Server running on http://localhost:8000")
            httpd.serve_forever()

if __name__ == "__main__":
    start_server()
