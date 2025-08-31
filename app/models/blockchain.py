"""
Blockchain Verification System for Agricultural MRV Data
SHA-256 based proof-of-work blockchain for immutable record storage
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Block:
    """Individual block in the blockchain"""
    
    def __init__(self, index: int, transactions: List[Dict], previous_hash: str, 
                 timestamp: Optional[float] = None):
        """
        Initialize a new block
        
        Args:
            index: Block number in chain
            transactions: List of MRV transactions
            previous_hash: Hash of previous block
            timestamp: Block creation time
        """
        self.index = index
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.timestamp = timestamp or time.time()
        self.nonce = 0
        self.hash = None
        
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of block"""
        block_string = json.dumps({
            'index': self.index,
            'transactions': self.transactions,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'nonce': self.nonce
        }, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()
    
    def mine_block(self, difficulty: int = 4) -> None:
        """
        Mine block using proof-of-work
        
        Args:
            difficulty: Number of leading zeros required
        """
        target = "0" * difficulty
        start_time = time.time()
        
        logger.info(f"Mining block {self.index} with difficulty {difficulty}")
        
        while True:
            self.hash = self.calculate_hash()
            if self.hash.startswith(target):
                break
            self.nonce += 1
            
            # Progress indicator every 100k attempts
            if self.nonce % 100000 == 0:
                logger.debug(f"Mining attempt {self.nonce}: {self.hash[:10]}")
        
        mining_time = time.time() - start_time
        logger.info(f"Block {self.index} mined! Hash: {self.hash[:16]}... "
                   f"(Nonce: {self.nonce}, Time: {mining_time:.2f}s)")
    
    def to_dict(self) -> Dict:
        """Convert block to dictionary"""
        return {
            'index': self.index,
            'transactions': self.transactions,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'nonce': self.nonce,
            'hash': self.hash
        }

class MRVTransaction:
    """MRV data transaction for blockchain storage"""
    
    def __init__(self, farm_id: str, mrv_data: Dict, transaction_type: str = "DATA_RECORD"):
        """
        Create MRV transaction
        
        Args:
            farm_id: Farm identifier
            mrv_data: Complete MRV data record
            transaction_type: Type of transaction
        """
        self.farm_id = farm_id
        self.mrv_data = mrv_data
        self.transaction_type = transaction_type
        self.timestamp = time.time()
        self.transaction_id = self._generate_transaction_id()
    
    def _generate_transaction_id(self) -> str:
        """Generate unique transaction ID"""
        tx_string = f"{self.farm_id}{self.timestamp}{self.transaction_type}".encode()
        return hashlib.sha256(tx_string).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        """Convert transaction to dictionary"""
        return {
            'transaction_id': self.transaction_id,
            'farm_id': self.farm_id,
            'transaction_type': self.transaction_type,
            'timestamp': self.timestamp,
            'mrv_data': self.mrv_data,
            'data_hash': hashlib.sha256(json.dumps(self.mrv_data, sort_keys=True).encode()).hexdigest()
        }

class MRVBlockchain:
    """Blockchain for MRV data verification and storage"""
    
    def __init__(self, difficulty: int = 3):
        """
        Initialize blockchain
        
        Args:
            difficulty: Mining difficulty (number of leading zeros)
        """
        self.chain = []
        self.pending_transactions = []
        self.mining_reward = 1.0  # Carbon credits for mining
        self.difficulty = difficulty
        
        # Create genesis block
        self._create_genesis_block()
        
        logger.info("MRV Blockchain initialized with genesis block")
    
    def _create_genesis_block(self) -> None:
        """Create the first block in chain"""
        genesis_transactions = [{
            'transaction_id': 'GENESIS',
            'farm_id': 'SYSTEM',
            'transaction_type': 'GENESIS',
            'timestamp': time.time(),
            'mrv_data': {
                'message': 'AgroMRV Blockchain Genesis Block',
                'created_for': 'NABARD Hackathon 2025',
                'purpose': 'Smallholder Farmer Climate Finance'
            },
            'data_hash': 'genesis'
        }]
        
        genesis_block = Block(0, genesis_transactions, "0")
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """Get the most recent block"""
        return self.chain[-1]
    
    def add_transaction(self, transaction: MRVTransaction) -> str:
        """
        Add transaction to pending pool
        
        Args:
            transaction: MRV transaction to add
            
        Returns:
            Transaction ID
        """
        self.pending_transactions.append(transaction.to_dict())
        logger.info(f"Added transaction {transaction.transaction_id} to pending pool")
        return transaction.transaction_id
    
    def mine_pending_transactions(self, mining_reward_address: str = "SYSTEM") -> Block:
        """
        Mine all pending transactions into new block
        
        Args:
            mining_reward_address: Address to receive mining reward
            
        Returns:
            Newly mined block
        """
        if not self.pending_transactions:
            logger.warning("No pending transactions to mine")
            return None
        
        # Add mining reward transaction
        reward_transaction = {
            'transaction_id': f"REWARD_{int(time.time())}",
            'farm_id': mining_reward_address,
            'transaction_type': 'MINING_REWARD',
            'timestamp': time.time(),
            'mrv_data': {
                'carbon_credits_earned': self.mining_reward,
                'block_index': len(self.chain)
            },
            'data_hash': 'reward'
        }
        
        transactions = self.pending_transactions + [reward_transaction]
        
        # Create and mine new block
        new_block = Block(
            len(self.chain),
            transactions,
            self.get_latest_block().hash
        )
        
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        
        # Clear pending transactions
        self.pending_transactions = []
        
        logger.info(f"Successfully mined block {new_block.index} with {len(transactions)} transactions")
        return new_block
    
    def is_chain_valid(self) -> bool:
        """Validate entire blockchain integrity"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if current block hash is valid
            if current_block.hash != current_block.calculate_hash():
                logger.error(f"Invalid hash at block {i}")
                return False
            
            # Check if previous hash matches
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Invalid previous hash at block {i}")
                return False
        
        return True
    
    def get_farm_transactions(self, farm_id: str) -> List[Dict]:
        """Get all transactions for a specific farm"""
        farm_transactions = []
        
        for block in self.chain:
            for transaction in block.transactions:
                if transaction.get('farm_id') == farm_id:
                    farm_transactions.append({
                        **transaction,
                        'block_index': block.index,
                        'block_hash': block.hash,
                        'block_timestamp': block.timestamp
                    })
        
        return farm_transactions
    
    def get_carbon_credit_balance(self, farm_id: str) -> float:
        """Calculate total carbon credits for a farm"""
        transactions = self.get_farm_transactions(farm_id)
        total_credits = 0.0
        
        for tx in transactions:
            if tx['transaction_type'] == 'DATA_RECORD':
                mrv_data = tx.get('mrv_data', {})
                credits = mrv_data.get('carbon_credits_potential', 0)
                total_credits += credits
            elif tx['transaction_type'] == 'MINING_REWARD':
                total_credits += tx['mrv_data'].get('carbon_credits_earned', 0)
        
        return round(total_credits, 4)
    
    def generate_carbon_certificate(self, farm_id: str) -> Dict:
        """Generate carbon credit certificate for farm"""
        transactions = self.get_farm_transactions(farm_id)
        
        if not transactions:
            return None
        
        total_credits = self.get_carbon_credit_balance(farm_id)
        total_co2_sequestered = sum(
            tx['mrv_data'].get('co2_sequestered_kg', 0) 
            for tx in transactions 
            if tx['transaction_type'] == 'DATA_RECORD'
        )
        
        certificate = {
            'certificate_id': hashlib.sha256(f"{farm_id}{time.time()}".encode()).hexdigest()[:16],
            'farm_id': farm_id,
            'issued_date': datetime.now().isoformat(),
            'total_carbon_credits': total_credits,
            'total_co2_sequestered_kg': round(total_co2_sequestered, 2),
            'verification_method': 'Blockchain + AI/ML Verification',
            'blockchain_verified': self.is_chain_valid(),
            'transaction_count': len(transactions),
            'latest_block_hash': self.get_latest_block().hash,
            'certificate_authority': 'AgroMRV System - NABARD 2025',
            'validity_period': '1 Year',
            'compliance_standard': 'IPCC Tier 2 Guidelines'
        }
        
        return certificate
    
    def get_blockchain_stats(self) -> Dict:
        """Get comprehensive blockchain statistics"""
        total_transactions = sum(len(block.transactions) for block in self.chain)
        unique_farms = set()
        total_carbon_credits = 0.0
        
        for block in self.chain:
            for tx in block.transactions:
                if tx['transaction_type'] == 'DATA_RECORD':
                    unique_farms.add(tx['farm_id'])
                    total_carbon_credits += tx['mrv_data'].get('carbon_credits_potential', 0)
        
        return {
            'total_blocks': len(self.chain),
            'total_transactions': total_transactions,
            'unique_farms': len(unique_farms),
            'total_carbon_credits': round(total_carbon_credits, 4),
            'blockchain_valid': self.is_chain_valid(),
            'mining_difficulty': self.difficulty,
            'latest_block_hash': self.get_latest_block().hash,
            'chain_creation_time': datetime.fromtimestamp(self.chain[0].timestamp).isoformat()
        }
    
    def export_chain(self) -> List[Dict]:
        """Export entire blockchain as JSON-serializable format"""
        return [block.to_dict() for block in self.chain]

class CarbonCreditRegistry:
    """Registry for managing carbon credit certificates"""
    
    def __init__(self, blockchain: MRVBlockchain):
        """
        Initialize registry with blockchain reference
        
        Args:
            blockchain: MRV blockchain instance
        """
        self.blockchain = blockchain
        self.certificates = {}
        
    def issue_certificate(self, farm_id: str) -> Optional[Dict]:
        """Issue new carbon credit certificate"""
        certificate = self.blockchain.generate_carbon_certificate(farm_id)
        
        if certificate:
            self.certificates[certificate['certificate_id']] = certificate
            logger.info(f"Issued certificate {certificate['certificate_id']} for farm {farm_id}")
            
        return certificate
    
    def verify_certificate(self, certificate_id: str) -> Dict:
        """Verify certificate authenticity"""
        certificate = self.certificates.get(certificate_id)
        
        if not certificate:
            return {'valid': False, 'error': 'Certificate not found'}
        
        # Verify blockchain integrity
        blockchain_valid = self.blockchain.is_chain_valid()
        
        # Check if farm transactions exist
        farm_transactions = self.blockchain.get_farm_transactions(certificate['farm_id'])
        
        return {
            'valid': blockchain_valid and len(farm_transactions) > 0,
            'blockchain_verified': blockchain_valid,
            'transaction_count': len(farm_transactions),
            'certificate_data': certificate
        }
    
    def get_all_certificates(self) -> List[Dict]:
        """Get all issued certificates"""
        return list(self.certificates.values())

def demo_blockchain_usage():
    """Demonstrate blockchain functionality"""
    
    # Initialize blockchain
    blockchain = MRVBlockchain(difficulty=2)  # Lower difficulty for demo
    registry = CarbonCreditRegistry(blockchain)
    
    # Sample MRV data
    sample_data = {
        'date': '2025-01-01',
        'co2_sequestered_kg': 125.5,
        'co2_emissions_kg': 45.2,
        'net_carbon_balance_kg': 80.3,
        'carbon_credits_potential': 0.289,
        'sustainability_score': 85.2
    }
    
    # Create and add transactions
    tx1 = MRVTransaction('FARM001', sample_data)
    tx2 = MRVTransaction('FARM002', {**sample_data, 'carbon_credits_potential': 0.156})
    
    blockchain.add_transaction(tx1)
    blockchain.add_transaction(tx2)
    
    # Mine block
    new_block = blockchain.mine_pending_transactions()
    
    # Issue certificates
    cert1 = registry.issue_certificate('FARM001')
    cert2 = registry.issue_certificate('FARM002')
    
    # Display results
    print("=== Blockchain Demo Results ===")
    print(f"Blockchain valid: {blockchain.is_chain_valid()}")
    print(f"Total blocks: {len(blockchain.chain)}")
    print(f"FARM001 carbon credits: {blockchain.get_carbon_credit_balance('FARM001')}")
    print(f"Certificate issued: {cert1['certificate_id'] if cert1 else 'None'}")
    
    return blockchain, registry

if __name__ == "__main__":
    demo_blockchain_usage()