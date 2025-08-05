// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title FinTechUserRegistry
 * @dev Smart contract for managing user profiles and financial transactions on blockchain
 */
contract FinTechUserRegistry {
    
    // Events
    event UserProfileCreated(string indexed userId, address indexed userAddress, uint256 timestamp);
    event TransactionAdded(string indexed userId, string indexed transactionType, uint256 timestamp);
    event UserProfileUpdated(string indexed userId, uint256 timestamp);
    
    // Structures
    struct UserProfile {
        string userId;
        string profileData; // JSON string containing user profile data
        address userAddress;
        uint256 createdAt;
        uint256 updatedAt;
        bool isActive;
    }
    
    struct Transaction {
        string transactionId;
        string userId;
        string transactionType; // LOAN_DECISION, FRAUD_ALERT, etc.
        string data; // JSON string containing transaction data
        uint256 timestamp;
        address createdBy;
    }
    
    // State variables
    mapping(string => UserProfile) public userProfiles;
    mapping(string => Transaction[]) public userTransactions;
    mapping(string => bool) public userExists;
    
    Transaction[] public allTransactions;
    string[] public allUserIds;
    
    address public owner;
    uint256 public totalUsers;
    uint256 public totalTransactions;
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier userNotExists(string memory _userId) {
        require(!userExists[_userId], "User already exists");
        _;
    }
    
    modifier userMustExist(string memory _userId) {
        require(userExists[_userId], "User does not exist");
        _;
    }
    
    // Constructor
    constructor() {
        owner = msg.sender;
        totalUsers = 0;
        totalTransactions = 0;
    }
    
    /**
     * @dev Create a new user profile
     * @param _userId Unique user identifier
     * @param _profileData JSON string containing user profile data
     */
    function createUserProfile(
        string memory _userId,
        string memory _profileData
    ) public userNotExists(_userId) {
        
        UserProfile memory newProfile = UserProfile({
            userId: _userId,
            profileData: _profileData,
            userAddress: msg.sender,
            createdAt: block.timestamp,
            updatedAt: block.timestamp,
            isActive: true
        });
        
        userProfiles[_userId] = newProfile;
        userExists[_userId] = true;
        allUserIds.push(_userId);
        totalUsers++;
        
        emit UserProfileCreated(_userId, msg.sender, block.timestamp);
    }
    
    /**
     * @dev Add a transaction for a user
     * @param _userId User identifier
     * @param _transactionType Type of transaction (LOAN_DECISION, FRAUD_ALERT, etc.)
     * @param _data JSON string containing transaction data
     */
    function addTransaction(
        string memory _userId,
        string memory _transactionType,
        string memory _data
    ) public {
        
        // Generate transaction ID
        string memory transactionId = generateTransactionId(_userId, _transactionType);
        
        Transaction memory newTransaction = Transaction({
            transactionId: transactionId,
            userId: _userId,
            transactionType: _transactionType,
            data: _data,
            timestamp: block.timestamp,
            createdBy: msg.sender
        });
        
        userTransactions[_userId].push(newTransaction);
        allTransactions.push(newTransaction);
        totalTransactions++;
        
        emit TransactionAdded(_userId, _transactionType, block.timestamp);
    }
    
    /**
     * @dev Get user profile data
     * @param _userId User identifier
     * @return profileData JSON string containing user profile data
     */
    function getUserProfile(string memory _userId) 
        public 
        view 
        userMustExist(_userId) 
        returns (string memory) 
    {
        return userProfiles[_userId].profileData;
    }
    
    /**
     * @dev Get all transactions for a user
     * @param _userId User identifier
     * @return Array of transaction data strings
     */
    function getUserTransactions(string memory _userId) 
        public 
        view 
        returns (string[] memory) 
    {
        Transaction[] memory transactions = userTransactions[_userId];
        string[] memory transactionData = new string[](transactions.length);
        
        for (uint256 i = 0; i < transactions.length; i++) {
            transactionData[i] = transactions[i].data;
        }
        
        return transactionData;
    }
    
    /**
     * @dev Get platform statistics
     * @return totalUsers, totalTransactions, contractBalance
     */
    function getPlatformStats() 
        public 
        view 
        returns (uint256, uint256, uint256) 
    {
        return (totalUsers, totalTransactions, address(this).balance);
    }
    
    /**
     * @dev Generate a unique transaction ID
     * @param _userId User identifier
     * @param _transactionType Type of transaction
     * @return Generated transaction ID
     */
    function generateTransactionId(
        string memory _userId, 
        string memory _transactionType
    ) private view returns (string memory) {
        return string(abi.encodePacked(
            _transactionType,
            "_",
            _userId,
            "_",
            uint2str(block.timestamp),
            "_",
            uint2str(totalTransactions)
        ));
    }
    
    /**
     * @dev Convert uint to string
     * @param _i Integer to convert
     * @return String representation of the integer
     */
    function uint2str(uint256 _i) private pure returns (string memory) {
        if (_i == 0) {
            return "0";
        }
        uint256 j = _i;
        uint256 len;
        while (j != 0) {
            len++;
            j /= 10;
        }
        bytes memory bstr = new bytes(len);
        uint256 k = len;
        while (_i != 0) {
            k = k - 1;
            uint8 temp = (48 + uint8(_i - _i / 10 * 10));
            bytes1 b1 = bytes1(temp);
            bstr[k] = b1;
            _i /= 10;
        }
        return string(bstr);
    }
    
    /**
     * @dev Transfer ownership (only current owner)
     * @param _newOwner Address of the new owner
     */
    function transferOwnership(address _newOwner) public onlyOwner {
        require(_newOwner != address(0), "New owner cannot be zero address");
        owner = _newOwner;
    }
    
    /**
     * @dev Receive Ether
     */
    receive() external payable {}
}
