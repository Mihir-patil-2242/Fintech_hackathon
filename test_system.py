#!/usr/bin/env python3
"""
FinTech AI Platform - System Verification Script
Tests all API endpoints and functionality
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_health():
    """Test system health"""
    print("🏥 Testing system health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System healthy - {len(data['services'])} services active")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_loan_application():
    """Test loan application with AI"""
    print("\n💰 Testing loan application...")
    try:
        loan_data = {
            "user_id": "test_user_001",
            "income": 600000,
            "loan_amount": 250000,
            "credit_history_score": 7,
            "employment_years": 5,
            "existing_debts": 75000,
            "purpose": "business"
        }
        
        response = requests.post(f"{BASE_URL}/loan-request", json=loan_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Loan processed - Approved: {data['approved']}, Credit Score: {data['credit_score']}")
            print(f"   Reason: {data['reason']}")
            if data.get('blockchain_tx_id'):
                print(f"   Blockchain TX: {data['blockchain_tx_id'][:16]}...")
            return True
        else:
            print(f"❌ Loan application failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Loan application error: {e}")
        return False

def test_fraud_detection():
    """Test fraud detection"""
    print("\n🚨 Testing fraud detection...")
    try:
        fraud_data = {
            "transaction_id": "tx_test_001",
            "user_id": "test_user_001",
            "amount": 150000,
            "merchant": "suspicious_merchant",
            "location": "unknown_location",
            "time_of_day": 3  # 3 AM - suspicious time
        }
        
        response = requests.post(f"{BASE_URL}/check-fraud", json=fraud_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Fraud check completed - Risk: {data['fraud_probability']:.2f}, Fraud: {data['is_fraud']}")
            return True
        else:
            print(f"❌ Fraud detection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Fraud detection error: {e}")
        return False

def test_market_data():
    """Test market data and investment advice"""
    print("\n📈 Testing market data...")
    try:
        response = requests.get(f"{BASE_URL}/market-data/popular")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Market data fetched - {data['count']} stocks analyzed")
            for stock in data['data'][:2]:  # Show first 2
                print(f"   {stock['symbol']}: ₹{stock['current_price']} - {stock['recommendation']}")
            return True
        else:
            print(f"❌ Market data failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Market data error: {e}")
        return False

def test_chatbot():
    """Test AI chatbot"""
    print("\n🤖 Testing AI chatbot...")
    try:
        chat_data = {
            "user_id": "test_user_001",
            "message": "What's my credit score?"
        }
        
        response = requests.post(f"{BASE_URL}/chatbot", json=chat_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chatbot responded - Confidence: {data['confidence']:.2f}")
            print(f"   Response: {data['response'][:100]}...")
            return True
        else:
            print(f"❌ Chatbot failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Chatbot error: {e}")
        return False

def test_blockchain():
    """Test blockchain functionality"""
    print("\n🔗 Testing blockchain...")
    try:
        # Get blockchain stats
        response = requests.get(f"{BASE_URL}/blockchain/stats")
        if response.status_code == 200:
            data = response.json()
            stats = data['blockchain_stats']
            print(f"✅ Blockchain active - {stats['total_blocks']} blocks, Valid: {stats['is_valid']}")
            
            # Get recent transactions
            tx_response = requests.get(f"{BASE_URL}/transactions?limit=5")
            if tx_response.status_code == 200:
                tx_data = tx_response.json()
                print(f"   Recent transactions: {tx_data['count']} found")
                return True
            else:
                print(f"❌ Transaction retrieval failed: {tx_response.status_code}")
                return False
        else:
            print(f"❌ Blockchain stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Blockchain error: {e}")
        return False

def test_compliance():
    """Test compliance checking"""
    print("\n⚖️ Testing compliance check...")
    try:
        compliance_data = {
            "user_id": "test_user_001",
            "transaction_amount": 1200000,  # High amount
            "kyc_completed": False,  # Missing KYC
            "source_of_funds": "cash"  # Suspicious source
        }
        
        response = requests.post(f"{BASE_URL}/compliance-check", json=compliance_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Compliance checked - Compliant: {data['compliant']}")
            if data['violations']:
                print(f"   Violations: {len(data['violations'])} found")
            return True
        else:
            print(f"❌ Compliance check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Compliance check error: {e}")
        return False

def main():
    """Run all system tests"""
    print("🏦 FinTech AI Platform - System Verification")
    print("=" * 50)
    print(f"Testing backend at: {BASE_URL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Health Check", test_health),
        ("Loan Application", test_loan_application),
        ("Fraud Detection", test_fraud_detection),
        ("Market Data", test_market_data),
        ("AI Chatbot", test_chatbot),
        ("Blockchain", test_blockchain),
        ("Compliance", test_compliance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All systems operational! Platform ready for demo.")
        return True
    else:
        print("⚠️  Some tests failed. Check backend logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
