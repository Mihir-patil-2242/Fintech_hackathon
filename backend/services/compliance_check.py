from typing import Dict, Any, List, Tuple

class ComplianceService:
    def __init__(self):
        # AML/KYC rules and thresholds
        self.high_value_threshold = 1000000  # ₹10 lakh
        self.suspicious_keywords = [
            'cash', 'anonymous', 'offshore', 'shell', 'bearer',
            'cryptocurrency', 'bitcoin', 'gambling', 'casino'
        ]
        self.high_risk_countries = [
            'unknown', 'offshore', 'anonymous', 'bearer'
        ]
    
    def check_compliance(self, request_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check AML/KYC compliance for a transaction or user
        Returns: (is_compliant, list_of_violations)
        """
        violations = []
        
        user_id = request_data.get('user_id', '')
        transaction_amount = request_data.get('transaction_amount', 0)
        kyc_completed = request_data.get('kyc_completed', False)
        source_of_funds = request_data.get('source_of_funds', '').lower()
        
        # Check KYC completion
        if not kyc_completed:
            violations.append("KYC not completed")
        
        # Check high-value transaction
        if transaction_amount > self.high_value_threshold:
            violations.append(f"High-value transaction (₹{transaction_amount:,.0f}) requires additional verification")
        
        # Check source of funds
        if self._is_suspicious_source(source_of_funds):
            violations.append(f"Suspicious source of funds: {source_of_funds}")
        
        # Check for suspicious patterns
        suspicious_patterns = self._check_suspicious_patterns(request_data)
        violations.extend(suspicious_patterns)
        
        # Additional checks based on user profile
        user_violations = self._check_user_profile(request_data)
        violations.extend(user_violations)
        
        is_compliant = len(violations) == 0
        
        return is_compliant, violations
    
    def _is_suspicious_source(self, source_of_funds: str) -> bool:
        """Check if source of funds contains suspicious keywords"""
        source_lower = source_of_funds.lower()
        return any(keyword in source_lower for keyword in self.suspicious_keywords)
    
    def _check_suspicious_patterns(self, request_data: Dict[str, Any]) -> List[str]:
        """Check for suspicious transaction patterns"""
        violations = []
        
        # Check for round number amounts (often suspicious)
        amount = request_data.get('transaction_amount', 0)
        if amount > 0 and amount % 100000 == 0 and amount >= 500000:
            violations.append("Round number high-value transaction may require scrutiny")
        
        # Check transaction frequency (simplified - in real system would check database)
        daily_transaction_count = request_data.get('daily_transaction_count', 0)
        if daily_transaction_count > 10:
            violations.append("High frequency transactions detected")
        
        # Check for structuring (amounts just below reporting thresholds)
        if 900000 <= amount < 1000000:
            violations.append("Potential structuring - amount just below reporting threshold")
        
        return violations
    
    def _check_user_profile(self, request_data: Dict[str, Any]) -> List[str]:
        """Check user profile for compliance issues"""
        violations = []
        
        # Check if user is on any watchlist (simplified)
        user_id = request_data.get('user_id', '')
        if self._is_on_watchlist(user_id):
            violations.append("User is on compliance watchlist")
        
        # Check beneficial ownership disclosure
        beneficial_owner_disclosed = request_data.get('beneficial_owner_disclosed', True)
        transaction_amount = request_data.get('transaction_amount', 0)
        
        if transaction_amount > 500000 and not beneficial_owner_disclosed:
            violations.append("Beneficial ownership not disclosed for high-value transaction")
        
        # Check PEP (Politically Exposed Person) status
        is_pep = request_data.get('is_pep', False)
        if is_pep and transaction_amount > 200000:
            violations.append("PEP high-value transaction requires enhanced due diligence")
        
        return violations
    
    def _is_on_watchlist(self, user_id: str) -> bool:
        """Check if user is on compliance watchlist (mock implementation)"""
        # In a real system, this would check against actual watchlists
        suspicious_user_patterns = ['test_suspicious', 'fraud_user', 'watchlist_']
        return any(pattern in user_id.lower() for pattern in suspicious_user_patterns)
    
    def generate_compliance_report(self, violations: List[str]) -> Dict[str, Any]:
        """Generate a compliance report based on violations"""
        if not violations:
            return {
                'status': 'COMPLIANT',
                'risk_level': 'LOW',
                'action_required': False,
                'message': 'Transaction complies with all AML/KYC requirements'
            }
        
        # Determine risk level based on number and type of violations
        risk_level = self._calculate_risk_level(violations)
        
        return {
            'status': 'NON_COMPLIANT',
            'risk_level': risk_level,
            'action_required': True,
            'violations': violations,
            'recommended_actions': self._get_recommended_actions(violations),
            'message': f'{len(violations)} compliance violation(s) detected'
        }
    
    def _calculate_risk_level(self, violations: List[str]) -> str:
        """Calculate risk level based on violations"""
        high_risk_keywords = ['watchlist', 'pep', 'suspicious', 'structuring']
        
        high_risk_count = sum(1 for violation in violations 
                             if any(keyword in violation.lower() for keyword in high_risk_keywords))
        
        if high_risk_count > 0:
            return 'HIGH'
        elif len(violations) > 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_recommended_actions(self, violations: List[str]) -> List[str]:
        """Get recommended actions based on violations"""
        actions = []
        
        violation_text = ' '.join(violations).lower()
        
        if 'kyc not completed' in violation_text:
            actions.append('Complete KYC verification immediately')
        
        if 'high-value' in violation_text:
            actions.append('Obtain additional documentation for high-value transaction')
        
        if 'suspicious' in violation_text:
            actions.append('Conduct enhanced due diligence')
        
        if 'watchlist' in violation_text:
            actions.append('Escalate to compliance officer immediately')
        
        if 'pep' in violation_text:
            actions.append('Apply enhanced PEP monitoring procedures')
        
        if 'structuring' in violation_text:
            actions.append('Investigate for potential money laundering')
        
        if not actions:
            actions.append('Review and document compliance decision')
        
        return actions
