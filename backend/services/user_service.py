import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json
import os
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from models.loan_request import UserRegistration
from services.auth_service import AuthService

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fintech_users.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    phone_number = Column(String, nullable=True)
    date_of_birth = Column(String, nullable=True)
    address = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    is_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    blockchain_tx_id = Column(String, nullable=True)
    kyc_level = Column(Integer, default=0)
    risk_score = Column(Float, default=0.5)

# Create tables
Base.metadata.create_all(bind=engine)

class UserService:
    def __init__(self):
        self.auth_service = AuthService()
    
    def get_db(self) -> Session:
        """Get database session"""
        db = SessionLocal()
        try:
            return db
        finally:
            pass  # Don't close here, let caller handle it
    
    async def create_user(self, user_data: UserRegistration) -> User:
        """Create a new user"""
        db = self.get_db()
        try:
            user_id = f"user_{uuid.uuid4().hex[:12]}"
            hashed_password = self.auth_service.hash_password(user_data.password)
            
            db_user = User(
                user_id=user_id,
                email=user_data.email,
                hashed_password=hashed_password,
                full_name=user_data.full_name,
                phone_number=user_data.phone_number,
                date_of_birth=user_data.date_of_birth,
                address=user_data.address
            )
            
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            
            return db_user
        finally:
            db.close()
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        db = self.get_db()
        try:
            return db.query(User).filter(User.email == email).first()
        finally:
            db.close()
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        db = self.get_db()
        try:
            return db.query(User).filter(User.user_id == user_id).first()
        finally:
            db.close()
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user credentials"""
        user = await self.get_user_by_email(email)
        if not user:
            return None
        
        if not self.auth_service.verify_password(password, user.hashed_password):
            return None
        
        return user
    
    async def update_user_blockchain_tx(self, user_id: str, tx_id: str) -> bool:
        """Update user's blockchain transaction ID"""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.user_id == user_id).first()
            if user:
                user.blockchain_tx_id = tx_id
                user.updated_at = datetime.utcnow()
                db.commit()
                return True
            return False
        finally:
            db.close()
    
    async def get_user_transaction_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's transaction history (mock implementation)"""
        # In a real implementation, this would query transaction database
        # For now, return mock data based on user_id
        mock_history = [
            {
                "transaction_id": f"tx_{user_id}_{i}",
                "amount": 1000.0 * (i + 1),
                "type": "payment" if i % 2 == 0 else "transfer",
                "timestamp": (datetime.utcnow() - timedelta(days=i*7)).isoformat(),
                "status": "completed"
            }
            for i in range(5)  # Last 5 transactions
        ]
        return mock_history
    
    async def get_account_age(self, user_id: str) -> int:
        """Get account age in days"""
        user = await self.get_user_by_id(user_id)
        if user:
            age = datetime.utcnow() - user.created_at
            return age.days
        return 0
    
    async def update_user_risk_score(self, user_id: str, risk_score: float) -> bool:
        """Update user's risk score"""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.user_id == user_id).first()
            if user:
                user.risk_score = risk_score
                user.updated_at = datetime.utcnow()
                db.commit()
                return True
            return False
        finally:
            db.close()
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics"""
        user = await self.get_user_by_id(user_id)
        if not user:
            return {}
        
        transaction_history = await self.get_user_transaction_history(user_id)
        account_age = await self.get_account_age(user_id)
        
        return {
            "user_id": user_id,
            "account_age_days": account_age,
            "total_transactions": len(transaction_history),
            "risk_score": user.risk_score,
            "kyc_level": user.kyc_level,
            "is_verified": user.is_verified,
            "last_login": user.updated_at.isoformat() if user.updated_at else None
        }
    
    async def search_users(self, query: str, limit: int = 10) -> List[User]:
        """Search users by name or email (admin function)"""
        db = self.get_db()
        try:
            return db.query(User).filter(
                (User.full_name.contains(query)) | (User.email.contains(query))
            ).limit(limit).all()
        finally:
            db.close()
    
    async def get_user_activity_summary(self, user_id: str) -> Dict[str, Any]:
        """Get user activity summary for dashboard"""
        user_stats = await self.get_user_stats(user_id)
        transaction_history = await self.get_user_transaction_history(user_id)
        
        # Calculate activity metrics
        total_volume = sum(tx.get("amount", 0) for tx in transaction_history)
        avg_transaction = total_volume / len(transaction_history) if transaction_history else 0
        
        return {
            **user_stats,
            "total_transaction_volume": total_volume,
            "average_transaction_amount": avg_transaction,
            "recent_activity_count": len([
                tx for tx in transaction_history 
                if datetime.fromisoformat(tx["timestamp"]) > datetime.utcnow() - timedelta(days=30)
            ])
        }
    
    async def health_check(self) -> str:
        """Health check for user service"""
        try:
            db = self.get_db()
            # Test database connection
            db.execute("SELECT 1")
            db.close()
            return "healthy"
        except Exception:
            return "unhealthy"
