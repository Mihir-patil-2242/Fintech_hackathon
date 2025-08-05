import hashlib
import json
import time
import sqlite3
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBlock:
    def __init__(self, index: int, timestamp: float, data: Dict[str, Any], 
                 previous_hash: str, nonce: int = 0):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
        self.merkle_root = self.calculate_merkle_root()
    
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the block"""
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'merkle_root': self.merkle_root if hasattr(self, 'merkle_root') else ''
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def calculate_merkle_root(self) -> str:
        """Calculate Merkle root of transactions in the block"""
        if not self.data or 'transactions' not in self.data:
            return hashlib.sha256(b'').hexdigest()
        
        transactions = self.data['transactions']
        if not transactions:
            return hashlib.sha256(b'').hexdigest()
        
        # Create leaf hashes
        hashes = []
        for tx in transactions:
            tx_string = json.dumps(tx, sort_keys=True)
            hashes.append(hashlib.sha256(tx_string.encode()).hexdigest())
        
        # Build Merkle tree
        while len(hashes) > 1:
            new_hashes = []
            for i in range(0, len(hashes), 2):
                left = hashes[i]
                right = hashes[i + 1] if i + 1 < len(hashes) else hashes[i]
                combined = left + right
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = new_hashes
        
        return hashes[0] if hashes else hashlib.sha256(b'').hexdigest()
    
    def mine_block(self, difficulty: int = 4):
        """Mine block using proof-of-work with adjustable difficulty"""
        target = "0" * difficulty
        start_time = time.time()
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
            
            # Progress logging every 100000 attempts
            if self.nonce % 100000 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Mining block {self.index}: {self.nonce} attempts, {elapsed:.1f}s")
        
        mining_time = time.time() - start_time
        logger.info(f"Block {self.index} mined! Hash: {self.hash}, Time: {mining_time:.2f}s, Nonce: {self.nonce}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary for storage"""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash,
            'merkle_root': self.merkle_root
        }
    
    @classmethod
    def from_dict(cls, block_dict: Dict[str, Any]) -> 'EnhancedBlock':
        """Create block from dictionary"""
        block = cls(
            block_dict['index'],
            block_dict['timestamp'],
            block_dict['data'],
            block_dict['previous_hash'],
            block_dict['nonce']
        )
        block.hash = block_dict['hash']
        block.merkle_root = block_dict.get('merkle_root', '')
        return block

class EnhancedBlockchain:
    def __init__(self, db_path: str = "blockchain.db"):
        self.db_path = db_path
        self.chain = []
        self.pending_transactions = []
        self.difficulty = 4
        self.mining_reward = 100
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Load existing blockchain or create genesis
        self._load_blockchain()
        
        # Transaction pool for better management
        self.transaction_pool = []
        self.max_transactions_per_block = 10
        
        # Network simulation
        self.peers = []
        self.is_mining = False
    
    def _init_database(self):
        """Initialize SQLite database for blockchain persistence"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create blocks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS blocks (
                    index INTEGER PRIMARY KEY,
                    timestamp REAL,
                    data TEXT,
                    previous_hash TEXT,
                    nonce INTEGER,
                    hash TEXT UNIQUE,
                    merkle_root TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create transactions table for better querying
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE,
                    block_index INTEGER,
                    type TEXT,
                    user_id TEXT,
                    data TEXT,
                    timestamp TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (block_index) REFERENCES blocks (index)
                )
            ''')
            
            # Create pending transactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pending_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE,
                    type TEXT,
                    user_id TEXT,
                    data TEXT,
                    timestamp TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def _load_blockchain(self):
        """Load blockchain from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM blocks ORDER BY index")
            rows = cursor.fetchall()
            
            if not rows:
                # Create genesis block
                self.chain = [self.create_genesis_block()]
                self._save_block_to_db(self.chain[0])
            else:
                # Load blocks from database
                self.chain = []
                for row in rows:
                    block_dict = {
                        'index': row[0],
                        'timestamp': row[1],
                        'data': json.loads(row[2]),
                        'previous_hash': row[3],
                        'nonce': row[4],
                        'hash': row[5],
                        'merkle_root': row[6] or ''
                    }
                    block = EnhancedBlock.from_dict(block_dict)
                    self.chain.append(block)
            
            # Load pending transactions
            cursor.execute("SELECT transaction_id, type, user_id, data, timestamp FROM pending_transactions")
            pending_rows = cursor.fetchall()
            
            for row in pending_rows:
                transaction = {
                    'transaction_id': row[0],
                    'type': row[1],
                    'user_id': row[2],
                    **json.loads(row[3]),
                    'timestamp': row[4]
                }
                self.pending_transactions.append(transaction)
            
            conn.close()
            logger.info(f"Loaded blockchain with {len(self.chain)} blocks and {len(self.pending_transactions)} pending transactions")
            
        except Exception as e:
            logger.error(f"Failed to load blockchain: {e}")
            self.chain = [self.create_genesis_block()]
    
    def _save_block_to_db(self, block: EnhancedBlock):
        """Save block to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO blocks 
                (index, timestamp, data, previous_hash, nonce, hash, merkle_root)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                block.index,
                block.timestamp,
                json.dumps(block.data),
                block.previous_hash,
                block.nonce,
                block.hash,
                block.merkle_root
            ))
            
            # Save transactions from this block
            if 'transactions' in block.data:
                for tx in block.data['transactions']:
                    cursor.execute('''
                        INSERT OR REPLACE INTO transactions 
                        (transaction_id, block_index, type, user_id, data, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        tx.get('transaction_id'),
                        block.index,
                        tx.get('type'),
                        tx.get('user_id'),
                        json.dumps(tx),
                        tx.get('timestamp')
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save block to database: {e}")
    
    def create_genesis_block(self) -> EnhancedBlock:
        """Create the first block in the chain"""
        genesis_data = {
            "message": "FinTech AI Platform Genesis Block",
            "transactions": [],
            "platform": "Blockchain + AI FinTech",
            "version": "1.0.0"
        }
        return EnhancedBlock(0, time.time(), genesis_data, "0")
    
    def get_latest_block(self) -> EnhancedBlock:
        """Get the last block in the chain"""
        return self.chain[-1]
    
    def add_transaction(self, transaction: Dict[str, Any]) -> str:
        """Add a transaction to pending transactions with validation"""
        with self.lock:
            # Generate transaction ID if not provided
            if 'transaction_id' not in transaction:
                transaction_id = hashlib.sha256(
                    json.dumps(transaction, sort_keys=True).encode()
                ).hexdigest()
                transaction['transaction_id'] = transaction_id
            
            # Add timestamp if not provided
            if 'timestamp' not in transaction:
                transaction['timestamp'] = datetime.now().isoformat()
            
            # Validate transaction
            if self._validate_transaction(transaction):
                self.pending_transactions.append(transaction)
                self._save_pending_transaction_to_db(transaction)
                logger.info(f"Added transaction {transaction['transaction_id']} to pending pool")
                
                return transaction['transaction_id']
            else:
                logger.warning(f"Invalid transaction rejected: {transaction}")
                return None
    
    def _validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Validate transaction before adding to pending pool"""
        required_fields = ['type', 'timestamp']
        
        # Check required fields
        for field in required_fields:
            if field not in transaction:
                logger.warning(f"Transaction missing required field: {field}")
                return False
        
        # Validate transaction type
        valid_types = ['LOAN_DECISION', 'FRAUD_ALERT', 'COMPLIANCE_ALERT', 'CHAT_LOG', 'MARKET_DATA']
        if transaction['type'] not in valid_types:
            logger.warning(f"Invalid transaction type: {transaction['type']}")
            return False
        
        # Type-specific validation
        if transaction['type'] == 'LOAN_DECISION':
            required = ['user_id', 'loan_id', 'approved', 'credit_score']
            if not all(field in transaction for field in required):
                return False
        
        elif transaction['type'] == 'FRAUD_ALERT':
            required = ['user_id', 'fraud_probability', 'is_fraud']
            if not all(field in transaction for field in required):
                return False
        
        return True
    
    def _save_pending_transaction_to_db(self, transaction: Dict[str, Any]):
        """Save pending transaction to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO pending_transactions 
                (transaction_id, type, user_id, data, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                transaction['transaction_id'],
                transaction['type'],
                transaction.get('user_id'),
                json.dumps(transaction),
                transaction['timestamp']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save pending transaction: {e}")
    
    def mine_pending_transactions(self) -> Optional[str]:
        """Mine pending transactions into a new block"""
        with self.lock:
            if not self.pending_transactions or self.is_mining:
                return None
            
            self.is_mining = True
            
            try:
                # Take transactions for mining
                transactions_to_mine = self.pending_transactions[:self.max_transactions_per_block]
                
                # Create new block
                block = EnhancedBlock(
                    len(self.chain),
                    time.time(),
                    {
                        'transactions': transactions_to_mine.copy(),
                        'count': len(transactions_to_mine),
                        'miner': 'FinTech-AI-Platform',
                        'reward': self.mining_reward
                    },
                    self.get_latest_block().hash
                )
                
                logger.info(f"Starting to mine block {block.index} with {len(transactions_to_mine)} transactions...")
                
                # Mine the block (this is CPU intensive)
                block.mine_block(self.difficulty)
                
                # Add to chain
                self.chain.append(block)
                
                # Save to database
                self._save_block_to_db(block)
                
                # Remove mined transactions from pending
                mined_tx_ids = [tx['transaction_id'] for tx in transactions_to_mine]
                self.pending_transactions = [
                    tx for tx in self.pending_transactions 
                    if tx['transaction_id'] not in mined_tx_ids
                ]
                
                # Remove from database
                self._remove_pending_transactions_from_db(mined_tx_ids)
                
                # Adjust difficulty based on mining time
                self._adjust_difficulty()
                
                logger.info(f"Block {block.index} successfully mined and added to chain")
                return block.hash
                
            except Exception as e:
                logger.error(f"Mining failed: {e}")
                return None
            finally:
                self.is_mining = False
    
    def _remove_pending_transactions_from_db(self, transaction_ids: List[str]):
        """Remove pending transactions from database after mining"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in transaction_ids])
            cursor.execute(f"DELETE FROM pending_transactions WHERE transaction_id IN ({placeholders})", transaction_ids)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to remove pending transactions: {e}")
    
    def _adjust_difficulty(self):
        """Adjust mining difficulty based on block time"""
        if len(self.chain) < 2:
            return
        
        # Calculate time for last block
        last_block = self.chain[-1]
        previous_block = self.chain[-2]
        block_time = last_block.timestamp - previous_block.timestamp
        
        # Target block time (e.g., 30 seconds)
        target_time = 30
        
        if block_time < target_time / 2:
            self.difficulty += 1
            logger.info(f"Increased difficulty to {self.difficulty}")
        elif block_time > target_time * 2:
            self.difficulty = max(1, self.difficulty - 1)
            logger.info(f"Decreased difficulty to {self.difficulty}")
    
    def is_chain_valid(self) -> bool:
        """Validate the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if current block's hash is valid
            if current_block.hash != current_block.calculate_hash():
                logger.error(f"Invalid hash at block {i}")
                return False
            
            # Check if current block points to previous block
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Invalid previous hash at block {i}")
                return False
            
            # Check proof of work
            if not current_block.hash.startswith("0" * self.difficulty):
                logger.error(f"Invalid proof of work at block {i}")
                return False
        
        return True
    
    def get_transaction_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent transactions from the blockchain"""
        transactions = []
        
        # Go through blocks in reverse order (newest first)
        for block in reversed(self.chain[1:]):  # Skip genesis block
            if 'transactions' in block.data:
                for tx in block.data['transactions']:
                    tx_with_block = tx.copy()
                    tx_with_block['block_index'] = block.index
                    tx_with_block['block_hash'] = block.hash
                    tx_with_block['confirmations'] = len(self.chain) - block.index
                    transactions.append(tx_with_block)
                    
                    if len(transactions) >= limit:
                        break
            
            if len(transactions) >= limit:
                break
        
        return transactions
    
    def get_user_transactions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all transactions for a specific user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT t.*, b.hash as block_hash 
                FROM transactions t 
                JOIN blocks b ON t.block_index = b.index 
                WHERE t.user_id = ? 
                ORDER BY t.timestamp DESC
            """, (user_id,))
            
            rows = cursor.fetchall()
            
            transactions = []
            for row in rows:
                tx_data = json.loads(row[5])  # data column
                tx_data['block_hash'] = row[8]  # block_hash
                tx_data['confirmations'] = len(self.chain) - row[2]  # block_index
                transactions.append(tx_data)
            
            conn.close()
            return transactions
            
        except Exception as e:
            logger.error(f"Failed to get user transactions: {e}")
            return []
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get comprehensive blockchain statistics"""
        return {
            'total_blocks': len(self.chain),
            'pending_transactions': len(self.pending_transactions),
            'is_valid': self.is_chain_valid(),
            'latest_block_hash': self.get_latest_block().hash,
            'difficulty': self.difficulty,
            'is_mining': self.is_mining,
            'total_transactions': sum(
                len(block.data.get('transactions', [])) 
                for block in self.chain[1:]  # Skip genesis
            ),
            'chain_size_mb': self._calculate_chain_size(),
            'average_block_time': self._calculate_average_block_time()
        }
    
    def _calculate_chain_size(self) -> float:
        """Calculate blockchain size in MB"""
        try:
            size_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            return round(size_bytes / (1024 * 1024), 2)
        except:
            return 0.0
    
    def _calculate_average_block_time(self) -> float:
        """Calculate average time between blocks"""
        if len(self.chain) < 2:
            return 0.0
        
        total_time = 0
        count = 0
        
        for i in range(1, min(len(self.chain), 11)):  # Last 10 blocks
            current = self.chain[-i]
            previous = self.chain[-i-1]
            total_time += current.timestamp - previous.timestamp
            count += 1
        
        return round(total_time / count, 2) if count > 0 else 0.0
    
    def get_block_info(self, block_index: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific block"""
        if 0 <= block_index < len(self.chain):
            block = self.chain[block_index]
            return {
                'index': block.index,
                'timestamp': block.timestamp,
                'hash': block.hash,
                'previous_hash': block.previous_hash,
                'merkle_root': block.merkle_root,
                'nonce': block.nonce,
                'data': block.data,
                'confirmations': len(self.chain) - block.index,
                'size_bytes': len(json.dumps(block.to_dict()))
            }
        return None

class EnhancedBlockchainService:
    def __init__(self, db_path: str = "fintech_blockchain.db"):
        self.blockchain = EnhancedBlockchain(db_path)
        self.auto_mine_enabled = True
        self.mining_thread = None
        
        # Start background mining if enabled
        if self.auto_mine_enabled:
            self._start_auto_mining()
    
    def _start_auto_mining(self):
        """Start background thread for automatic mining"""
        def auto_mine():
            while self.auto_mine_enabled:
                try:
                    if len(self.blockchain.pending_transactions) > 0:
                        self.blockchain.mine_pending_transactions()
                    time.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    logger.error(f"Auto-mining error: {e}")
                    time.sleep(30)  # Wait longer on error
        
        self.mining_thread = threading.Thread(target=auto_mine, daemon=True)
        self.mining_thread.start()
        logger.info("Auto-mining thread started")
    
    def add_loan_decision(self, loan_data: Dict[str, Any]) -> str:
        """Add loan decision to blockchain"""
        transaction = {
            'type': 'LOAN_DECISION',
            'user_id': loan_data.get('user_id'),
            'loan_id': loan_data.get('loan_id'),
            'approved': loan_data.get('approved'),
            'credit_score': loan_data.get('credit_score'),
            'amount': loan_data.get('loan_amount'),
            'reason': loan_data.get('reason'),
            'fraud_probability': loan_data.get('fraud_probability', 0),
            'processing_time_ms': loan_data.get('processing_time_ms', 0)
        }
        
        return self.blockchain.add_transaction(transaction)
    
    def add_fraud_alert(self, fraud_data: Dict[str, Any]) -> str:
        """Add fraud alert to blockchain"""
        transaction = {
            'type': 'FRAUD_ALERT',
            'transaction_id': fraud_data.get('transaction_id'),
            'user_id': fraud_data.get('user_id'),
            'fraud_probability': fraud_data.get('fraud_probability'),
            'is_fraud': fraud_data.get('is_fraud'),
            'amount': fraud_data.get('amount'),
            'detection_method': fraud_data.get('detection_method', 'AI'),
            'risk_factors': fraud_data.get('risk_factors', [])
        }
        
        return self.blockchain.add_transaction(transaction)
    
    def add_compliance_alert(self, compliance_data: Dict[str, Any]) -> str:
        """Add compliance violation to blockchain"""
        transaction = {
            'type': 'COMPLIANCE_ALERT',
            'user_id': compliance_data.get('user_id'),
            'compliant': compliance_data.get('compliant'),
            'violations': compliance_data.get('violations'),
            'risk_level': compliance_data.get('risk_level'),
            'transaction_amount': compliance_data.get('transaction_amount'),
            'recommended_actions': compliance_data.get('recommended_actions', [])
        }
        
        return self.blockchain.add_transaction(transaction)
    
    def add_chat_log(self, chat_data: Dict[str, Any]) -> str:
        """Add chat log hash to blockchain for audit"""
        # Only store hash for privacy
        chat_hash = hashlib.sha256(
            json.dumps(chat_data, sort_keys=True).encode()
        ).hexdigest()
        
        transaction = {
            'type': 'CHAT_LOG',
            'user_id': chat_data.get('user_id'),
            'chat_hash': chat_hash,
            'timestamp': chat_data.get('timestamp'),
            'intent': chat_data.get('intent'),
            'confidence': chat_data.get('confidence'),
            'session_id': chat_data.get('session_id')
        }
        
        return self.blockchain.add_transaction(transaction)
    
    def get_recent_transactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent blockchain transactions"""
        return self.blockchain.get_transaction_history(limit)
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get comprehensive blockchain statistics"""
        return self.blockchain.get_blockchain_stats()
    
    def get_user_transactions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all transactions for a specific user"""
        return self.blockchain.get_user_transactions(user_id)
    
    def force_mine_block(self) -> Optional[str]:
        """Manually trigger block mining"""
        return self.blockchain.mine_pending_transactions()
    
    def validate_blockchain(self) -> bool:
        """Validate entire blockchain integrity"""
        return self.blockchain.is_chain_valid()
