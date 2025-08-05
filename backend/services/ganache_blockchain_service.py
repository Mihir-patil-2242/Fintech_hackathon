import os
import json
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GanacheBlockchainService:
    def __init__(self):
        # Ganache connection
        self.ganache_url = os.getenv('GANACHE_URL', 'http://127.0.0.1:7545')
        self.chain_id = int(os.getenv('GANACHE_CHAIN_ID', '1337'))
        self.private_key = os.getenv('PRIVATE_KEY')
        self.contract_address = os.getenv('CONTRACT_ADDRESS')
        
        # Initialize Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(self.ganache_url))
        
        # Add PoA middleware for Ganache
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Get account from private key
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address
        else:
            # Use first account from Ganache for development
            accounts = self.w3.eth.accounts
            if accounts:
                self.address = accounts[0]
                logger.warning("Using Ganache account - set PRIVATE_KEY in production")
            else:
                raise Exception("No accounts available and no private key set")
        
        # Smart contract ABI (simplified version)
        self.contract_abi = [
            {
                "inputs": [
                    {"name": "_userId", "type": "string"},
                    {"name": "_data", "type": "string"}
                ],
                "name": "createUserProfile",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "_userId", "type": "string"},
                    {"name": "_transactionType", "type": "string"},
                    {"name": "_data", "type": "string"}
                ],
                "name": "addTransaction",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "_userId", "type": "string"}],
                "name": "getUserProfile",
                "outputs": [{"name": "", "type": "string"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"name": "_userId", "type": "string"}],
                "name": "getUserTransactions",
                "outputs": [{"name": "", "type": "string[]"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        # Initialize contract
        self.contract = None
        if self.contract_address and self.w3.is_address(self.contract_address):
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Ganache blockchain service initialized")
        logger.info(f"Connected to: {self.ganache_url}")
        logger.info(f"Account: {self.address}")
        logger.info(f"Contract: {self.contract_address}")
    
    def is_connected(self) -> bool:
        """Check if connected to Ganache"""
        try:
            return self.w3.is_connected()
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False
    
    async def create_user_profile(self, user_data: Dict[str, Any]) -> str:
        """Create user profile on blockchain"""
        try:
            if not self.contract:
                # Fallback: just create a transaction hash
                return await self._create_mock_transaction("USER_PROFILE", user_data)
            
            def _create_profile():
                user_id = user_data.get('user_id')
                data_json = json.dumps(user_data)
                
                # Build transaction
                transaction = self.contract.functions.createUserProfile(
                    user_id, data_json
                ).build_transaction({
                    'from': self.address,
                    'nonce': self.w3.eth.get_transaction_count(self.address),
                    'gas': 2000000,
                    'gasPrice': self.w3.to_wei('20', 'gwei'),
                    'chainId': self.chain_id
                })
                
                # Sign and send transaction
                if self.private_key:
                    signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
                    tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
                else:
                    tx_hash = self.w3.eth.send_transaction(transaction)
                
                # Wait for confirmation
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                return receipt.transactionHash.hex()
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            tx_hash = await loop.run_in_executor(self.executor, _create_profile)
            
            logger.info(f"User profile created on blockchain: {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            return await self._create_mock_transaction("USER_PROFILE", user_data)
    
    async def add_loan_decision(self, loan_data: Dict[str, Any]) -> str:
        """Add loan decision to blockchain"""
        return await self._add_transaction("LOAN_DECISION", loan_data)
    
    async def add_fraud_alert(self, fraud_data: Dict[str, Any]) -> str:
        """Add fraud alert to blockchain"""
        return await self._add_transaction("FRAUD_ALERT", fraud_data)
    
    async def add_compliance_alert(self, compliance_data: Dict[str, Any]) -> str:
        """Add compliance alert to blockchain"""
        return await self._add_transaction("COMPLIANCE_ALERT", compliance_data)
    
    async def log_user_event(self, event_data: Dict[str, Any]) -> str:
        """Log user event to blockchain"""
        return await self._add_transaction("USER_EVENT", event_data)
    
    async def add_chat_log(self, chat_data: Dict[str, Any]) -> str:
        """Add chat log hash to blockchain"""
        # Only store hash for privacy
        chat_hash = {
            'user_id': chat_data.get('user_id'),
            'chat_hash': hash(json.dumps(chat_data, sort_keys=True)),
            'timestamp': chat_data.get('timestamp'),
            'intent': chat_data.get('intent')
        }
        return await self._add_transaction("CHAT_LOG", chat_hash)
    
    async def _add_transaction(self, transaction_type: str, data: Dict[str, Any]) -> str:
        """Add transaction to blockchain"""
        try:
            if not self.contract:
                return await self._create_mock_transaction(transaction_type, data)
            
            def _add_tx():
                user_id = data.get('user_id', 'system')
                data_json = json.dumps(data)
                
                transaction = self.contract.functions.addTransaction(
                    user_id, transaction_type, data_json
                ).build_transaction({
                    'from': self.address,
                    'nonce': self.w3.eth.get_transaction_count(self.address),
                    'gas': 2000000,
                    'gasPrice': self.w3.to_wei('20', 'gwei'),
                    'chainId': self.chain_id
                })
                
                if self.private_key:
                    signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
                    tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
                else:
                    tx_hash = self.w3.eth.send_transaction(transaction)
                
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                return receipt.transactionHash.hex()
            
            loop = asyncio.get_event_loop()
            tx_hash = await loop.run_in_executor(self.executor, _add_tx)
            
            logger.info(f"Transaction added to blockchain: {transaction_type} - {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"Error adding transaction: {e}")
            return await self._create_mock_transaction(transaction_type, data)
    
    async def _create_mock_transaction(self, tx_type: str, data: Dict[str, Any]) -> str:
        """Create mock transaction hash when blockchain is not available"""
        mock_hash = f"0x{hash(json.dumps(data, sort_keys=True)):x}"[-16:]
        logger.warning(f"Created mock transaction hash: {mock_hash}")
        return mock_hash
    
    async def get_user_blockchain_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile from blockchain"""
        try:
            if not self.contract:
                return None
            
            def _get_profile():
                result = self.contract.functions.getUserProfile(user_id).call()
                if result:
                    return json.loads(result)
                return None
            
            loop = asyncio.get_event_loop()
            profile = await loop.run_in_executor(self.executor, _get_profile)
            return profile
            
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    async def get_user_transactions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user transactions from blockchain"""
        try:
            if not self.contract:
                return []
            
            def _get_transactions():
                results = self.contract.functions.getUserTransactions(user_id).call()
                transactions = []
                for result in results:
                    try:
                        tx_data = json.loads(result)
                        transactions.append(tx_data)
                    except:
                        continue
                return transactions
            
            loop = asyncio.get_event_loop()
            transactions = await loop.run_in_executor(self.executor, _get_transactions)
            return transactions
            
        except Exception as e:
            logger.error(f"Error getting user transactions: {e}")
            return []
    
    async def get_recent_transactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent transactions from blockchain"""
        try:
            def _get_recent():
                latest_block = self.w3.eth.get_block('latest')
                transactions = []
                
                # Get transactions from recent blocks
                for i in range(min(limit, latest_block.number)):
                    block_number = latest_block.number - i
                    block = self.w3.eth.get_block(block_number, full_transactions=True)
                    
                    for tx in block.transactions:
                        if tx.to == self.contract_address:
                            # Decode transaction data
                            try:
                                receipt = self.w3.eth.get_transaction_receipt(tx.hash)
                                tx_data = {
                                    'transaction_id': tx.hash.hex(),
                                    'block_number': block_number,
                                    'timestamp': datetime.fromtimestamp(block.timestamp).isoformat(),
                                    'from_address': tx['from'],
                                    'gas_used': receipt.gasUsed,
                                    'status': receipt.status
                                }
                                transactions.append(tx_data)
                            except:
                                continue
                
                return transactions[:limit]
            
            loop = asyncio.get_event_loop()
            transactions = await loop.run_in_executor(self.executor, _get_recent)
            return transactions
            
        except Exception as e:
            logger.error(f"Error getting recent transactions: {e}")
            return []
    
    async def get_enhanced_blockchain_stats(self) -> Dict[str, Any]:
        """Get comprehensive blockchain statistics"""
        try:
            def _get_stats():
                latest_block = self.w3.eth.get_block('latest')
                return {
                    'latest_block_number': latest_block.number,
                    'latest_block_hash': latest_block.hash.hex(),
                    'chain_id': self.chain_id,
                    'gas_limit': latest_block.gasLimit,
                    'gas_used': latest_block.gasUsed,
                    'timestamp': latest_block.timestamp,
                    'transaction_count': len(latest_block.transactions),
                    'network': 'Ganache Testnet',
                    'is_connected': self.is_connected(),
                    'account_balance': self.w3.from_wei(
                        self.w3.eth.get_balance(self.address), 'ether'
                    ),
                    'contract_address': self.contract_address
                }
            
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(self.executor, _get_stats)
            return stats
            
        except Exception as e:
            logger.error(f"Error getting blockchain stats: {e}")
            return {
                'latest_block_number': 0,
                'latest_block_hash': '0x0',
                'chain_id': self.chain_id,
                'network': 'Ganache Testnet (Offline)',
                'is_connected': False,
                'error': str(e)
            }
    
    async def health_check(self) -> str:
        """Health check for blockchain service"""
        try:
            if self.is_connected():
                latest_block = self.w3.eth.get_block('latest')
                if latest_block:
                    return "healthy"
            return "unhealthy"
        except Exception:
            return "unhealthy"
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
