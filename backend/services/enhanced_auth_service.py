import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
import secrets
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class EnhancedAuthService:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.refresh_token_expire_days = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != token_type:
                raise jwt.InvalidTokenError(f"Invalid token type. Expected {token_type}")
            
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token has expired")
        except jwt.JWTError as e:
            raise Exception(f"Invalid token: {str(e)}")
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Create new access token from refresh token"""
        try:
            payload = self.verify_token(refresh_token, "refresh")
            
            # Create new access token with same user data
            new_access_token = self.create_access_token({
                "sub": payload.get("sub"),
                "email": payload.get("email"),
                "user_id": payload.get("user_id")
            })
            
            return {
                "access_token": new_access_token,
                "token_type": "bearer"
            }
            
        except Exception as e:
            raise Exception(f"Failed to refresh token: {str(e)}")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt with salt"""
        salt = bcrypt.gensalt(rounds=12)  # Increased rounds for better security
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def generate_password_reset_token(self, user_id: str) -> str:
        """Generate password reset token"""
        data = {
            "sub": user_id,
            "type": "password_reset",
            "exp": datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
        }
        return jwt.encode(data, self.secret_key, algorithm=self.algorithm)
    
    def verify_password_reset_token(self, token: str) -> Optional[str]:
        """Verify password reset token and return user_id"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if payload.get("type") != "password_reset":
                return None
            return payload.get("sub")
        except jwt.JWTError:
            return None
    
    def generate_email_verification_token(self, email: str) -> str:
        """Generate email verification token"""
        data = {
            "email": email,
            "type": "email_verification",
            "exp": datetime.utcnow() + timedelta(hours=24)  # 24 hour expiry
        }
        return jwt.encode(data, self.secret_key, algorithm=self.algorithm)
    
    def verify_email_verification_token(self, token: str) -> Optional[str]:
        """Verify email verification token and return email"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if payload.get("type") != "email_verification":
                return None
            return payload.get("email")
        except jwt.JWTError:
            return None
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        requirements = {
            "min_length": len(password) >= 8,
            "has_uppercase": any(c.isupper() for c in password),
            "has_lowercase": any(c.islower() for c in password),
            "has_digit": any(c.isdigit() for c in password),
            "has_special": any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        }
        
        is_strong = all(requirements.values())
        
        return {
            "is_strong": is_strong,
            "requirements": requirements,
            "score": sum(requirements.values()) / len(requirements)
        }
    
    async def health_check(self) -> str:
        """Health check for auth service"""
        try:
            # Test token creation and verification
            test_data = {"test": "data", "user_id": "test_user"}
            
            # Test access token
            access_token = self.create_access_token(test_data)
            decoded_access = self.verify_token(access_token, "access")
            
            # Test refresh token
            refresh_token = self.create_refresh_token(test_data)
            decoded_refresh = self.verify_token(refresh_token, "refresh")
            
            # Test password hashing
            test_password = "test_password_123"
            hashed = self.hash_password(test_password)
            password_valid = self.verify_password(test_password, hashed)
            
            if (decoded_access.get("test") == "data" and 
                decoded_refresh.get("test") == "data" and 
                password_valid):
                return "healthy"
            else:
                return "unhealthy"
                
        except Exception as e:
            logger.error(f"Auth service health check failed: {e}")
            return "unhealthy"
