import os
from web3 import Web3
import solcx

# Ensure the Solidity compiler (solc) matching our smart contract version is installed
try:
    # We install exactly 0.8.0 to match the pragma in our contract
    solcx.install_solc('0.8.0')
except Exception as e:
    print(f"Note: solc installation message: {e}")

class BlockchainLogger:
    def __init__(self, rpc_url="http://127.0.0.1:7545"):
        # Connect to Ganache
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Ganache at {rpc_url}. Is it running?")
        
        print(f"Successfully connected to Ganache at {rpc_url}")
        
        # Use the first account in Ganache to pay for the "gas" (transaction fees)
        self.w3.eth.default_account = self.w3.eth.accounts[0]
        
        self.contract_address = None
        self.contract_abi = None
        self.contract = None
        
        self.setup_contract()

    def setup_contract(self):
        """Compiles and deploys the smart contract to Ganache."""
        contract_path = os.path.join(os.path.dirname(__file__), 'DisasterAudit.sol')
        with open(contract_path, 'r') as file:
            contract_source_code = file.read()
            
        print("Compiling Smart Contract...")
        compiled_sol = solcx.compile_source(
            contract_source_code,
            output_values=['abi', 'bin'],
            solc_version='0.8.0'
        )
        
        # Extract compiled interface
        contract_id, contract_interface = compiled_sol.popitem()
        bytecode = contract_interface['bin']
        abi = contract_interface['abi']
        
        # Deploy contract
        print("Deploying Smart Contract to Ganache...")
        DisasterAudit = self.w3.eth.contract(abi=abi, bytecode=bytecode)
        
        # Submit transaction that deploys the contract
        tx_hash = DisasterAudit.constructor().transact()
        # Wait for the transaction to be mined
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        self.contract_address = tx_receipt.contractAddress
        self.contract_abi = abi
        
        # Create the contract instance we will use to log events
        self.contract = self.w3.eth.contract(
            address=self.contract_address, 
            abi=self.contract_abi
        )
        print(f"Contract deployed successfully at address: {self.contract_address}")

    def log_event(self, agent_id, action_type, location):
        """
        Logs an event to the blockchain.
        e.g. log_event("Agent_1", "RESCUE_SURVIVOR", "x:10, y:20")
        """
        try:
            print(f"[Blockchain] Logging event: {agent_id} -> {action_type} at {location}")
            # Call the logDisasterEvent function on the smart contract
            tx_hash = self.contract.functions.logDisasterEvent(
                agent_id, action_type, location
            ).transact()
            
            # Wait for it to be confirmed on the blockchain
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"[Blockchain] Event mined! Block number: {receipt.blockNumber}")
        except Exception as e:
            print(f"[Blockchain Error] Failed to log event: {e}")

    def get_agent_balance(self, agent_id):
        """Fetches the RescueToken balance for a specific agent."""
        if not self.contract:
            return 0
            
        try:
            return self.contract.functions.getAgentBalance(agent_id).call()
        except Exception as e:
            print(f"[Blockchain Error] Failed to get balance for {agent_id}: {e}")
            return 0

    def get_total_logs(self):
        """Fetches the total number of events logged so far."""
        return self.contract.functions.getLogsCount().call()

    def print_all_logs(self):
        """Retrieves and prints all recorded logs."""
        count = self.get_total_logs()
        print(f"--- Blockchain Audit Logs ({count} total) ---")
        for i in range(count):
            timestamp, agent, action, loc = self.contract.functions.getLog(i).call()
            print(f"Log {i}: Agent={agent} | Action={action} | Location={loc} | Timestamp={timestamp}")
        print("---------------------------------------")

if __name__ == "__main__":
    # Test the logger
    print("Initializing Blockchain Logger test...")
    logger = BlockchainLogger()
    
    # Simulate an AI agent finding a hazard
    logger.log_event("Agent_Scout_1", "FIRE_DETECTED", "x:45, y:12")
    
    # Simulate another agent rescuing someone
    logger.log_event("Agent_Medic_2", "SURVIVOR_RESCUED", "x:46, y:12")
    
    # Read the data back from the blockchain
    logger.print_all_logs()
