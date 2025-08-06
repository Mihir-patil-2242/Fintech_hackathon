// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract LendingPool is ReentrancyGuard, Ownable {
    using SafeMath for uint256;
    
    IERC20 public loanToken;
    
    struct Loan {
        address borrower;
        uint256 amount;
        uint256 collateral;
        uint256 interestRate; // Annual rate in basis points (100 = 1%)
        uint256 duration; // In seconds
        uint256 startTime;
        uint256 repaidAmount;
        bool isActive;
        bool isDefaulted;
        uint256 creditScore;
    }
    
    struct Investment {
        address investor;
        uint256 amount;
        uint256 startTime;
        uint256 apy; // Annual percentage yield in basis points
        bool isActive;
    }
    
    struct DebtCycleIndicator {
        uint256 totalDebt;
        uint256 totalCollateral;
        uint256 averageInterestRate;
        uint256 defaultRate;
        uint256 creditGrowthRate;
        uint256 timestamp;
    }
    
    mapping(uint256 => Loan) public loans;
    mapping(uint256 => Investment) public investments;
    mapping(address => uint256[]) public userLoans;
    mapping(address => uint256[]) public userInvestments;
    mapping(address => uint256) public creditScores;
    
    DebtCycleIndicator public currentCycle;
    
    uint256 public loanCounter;
    uint256 public investmentCounter;
    uint256 public minCollateralRatio = 150; // 150% collateralization
    uint256 public liquidationThreshold = 120; // 120% threshold for liquidation
    uint256 public baseCreditScore = 600;
    
    event LoanCreated(uint256 indexed loanId, address indexed borrower, uint256 amount, uint256 collateral);
    event LoanRepaid(uint256 indexed loanId, uint256 amount);
    event LoanLiquidated(uint256 indexed loanId, address liquidator);
    event InvestmentMade(uint256 indexed investmentId, address indexed investor, uint256 amount);
    event InvestmentWithdrawn(uint256 indexed investmentId, uint256 amount);
    event CreditScoreUpdated(address indexed user, uint256 newScore);
    
    constructor(address _loanToken) {
        loanToken = IERC20(_loanToken);
    }
    
    // Create a loan with collateral
    function createLoan(uint256 _amount, uint256 _duration) external payable nonReentrant {
        require(_amount > 0, "Invalid loan amount");
        require(msg.value >= _amount.mul(minCollateralRatio).div(100), "Insufficient collateral");
        
        uint256 userCreditScore = getCreditScore(msg.sender);
        uint256 interestRate = calculateInterestRate(userCreditScore, _amount, _duration);
        
        loanCounter++;
        loans[loanCounter] = Loan({
            borrower: msg.sender,
            amount: _amount,
            collateral: msg.value,
            interestRate: interestRate,
            duration: _duration,
            startTime: block.timestamp,
            repaidAmount: 0,
            isActive: true,
            isDefaulted: false,
            creditScore: userCreditScore
        });
        
        userLoans[msg.sender].push(loanCounter);
        
        // Transfer loan tokens to borrower
        require(loanToken.transfer(msg.sender, _amount), "Token transfer failed");
        
        // Update debt cycle indicators
        updateDebtCycle(_amount, msg.value, interestRate);
        
        emit LoanCreated(loanCounter, msg.sender, _amount, msg.value);
    }
    
    // Repay loan
    function repayLoan(uint256 _loanId, uint256 _amount) external nonReentrant {
        Loan storage loan = loans[_loanId];
        require(loan.isActive, "Loan not active");
        require(msg.sender == loan.borrower, "Not loan borrower");
        
        uint256 totalDue = calculateTotalDue(_loanId);
        uint256 repayAmount = _amount > totalDue ? totalDue : _amount;
        
        require(loanToken.transferFrom(msg.sender, address(this), repayAmount), "Transfer failed");
        
        loan.repaidAmount = loan.repaidAmount.add(repayAmount);
        
        if (loan.repaidAmount >= totalDue) {
            loan.isActive = false;
            // Return collateral
            payable(loan.borrower).transfer(loan.collateral);
            
            // Improve credit score for successful repayment
            updateCreditScore(loan.borrower, 50, true);
        }
        
        emit LoanRepaid(_loanId, repayAmount);
    }
    
    // Make an investment
    function invest(uint256 _amount) external nonReentrant {
        require(_amount > 0, "Invalid investment amount");
        require(loanToken.transferFrom(msg.sender, address(this), _amount), "Transfer failed");
        
        uint256 apy = calculateAPY(_amount);
        
        investmentCounter++;
        investments[investmentCounter] = Investment({
            investor: msg.sender,
            amount: _amount,
            startTime: block.timestamp,
            apy: apy,
            isActive: true
        });
        
        userInvestments[msg.sender].push(investmentCounter);
        
        emit InvestmentMade(investmentCounter, msg.sender, _amount);
    }
    
    // Withdraw investment with yields
    function withdrawInvestment(uint256 _investmentId) external nonReentrant {
        Investment storage investment = investments[_investmentId];
        require(investment.isActive, "Investment not active");
        require(investment.investor == msg.sender, "Not investor");
        
        uint256 yield = calculateYield(_investmentId);
        uint256 totalAmount = investment.amount.add(yield);
        
        investment.isActive = false;
        
        require(loanToken.transfer(msg.sender, totalAmount), "Transfer failed");
        
        emit InvestmentWithdrawn(_investmentId, totalAmount);
    }
    
    // Calculate interest rate based on credit score and loan parameters
    function calculateInterestRate(uint256 _creditScore, uint256 _amount, uint256 _duration) 
        public 
        pure 
        returns (uint256) 
    {
        // Base rate: 5%
        uint256 baseRate = 500;
        
        // Adjust based on credit score (300-850 range)
        uint256 creditAdjustment = 0;
        if (_creditScore < 600) {
            creditAdjustment = 1000; // +10%
        } else if (_creditScore < 700) {
            creditAdjustment = 500; // +5%
        } else if (_creditScore >= 750) {
            creditAdjustment = 0; // No adjustment for excellent credit
        }
        
        // Adjust based on loan amount (higher amounts = higher risk)
        uint256 amountAdjustment = _amount.div(10**18).mul(10); // 0.1% per token
        
        // Adjust based on duration (longer = higher risk)
        uint256 durationAdjustment = _duration.div(86400).mul(5); // 0.05% per day
        
        return baseRate.add(creditAdjustment).add(amountAdjustment).add(durationAdjustment);
    }
    
    // Calculate total amount due for a loan
    function calculateTotalDue(uint256 _loanId) public view returns (uint256) {
        Loan memory loan = loans[_loanId];
        uint256 timeElapsed = block.timestamp.sub(loan.startTime);
        uint256 interest = loan.amount.mul(loan.interestRate).mul(timeElapsed).div(365 days).div(10000);
        return loan.amount.add(interest);
    }
    
    // Calculate APY for investments based on pool performance
    function calculateAPY(uint256 _amount) public view returns (uint256) {
        // Base APY: 8%
        uint256 baseAPY = 800;
        
        // Adjust based on pool health and debt cycle
        uint256 healthAdjustment = 0;
        if (currentCycle.defaultRate < 100) { // Less than 1% default rate
            healthAdjustment = 200; // +2% bonus
        } else if (currentCycle.defaultRate > 500) { // More than 5% default rate
            healthAdjustment = 0; // Reduced APY
        }
        
        return baseAPY.add(healthAdjustment);
    }
    
    // Calculate yield for an investment
    function calculateYield(uint256 _investmentId) public view returns (uint256) {
        Investment memory investment = investments[_investmentId];
        uint256 timeElapsed = block.timestamp.sub(investment.startTime);
        return investment.amount.mul(investment.apy).mul(timeElapsed).div(365 days).div(10000);
    }
    
    // Get or initialize credit score
    function getCreditScore(address _user) public view returns (uint256) {
        uint256 score = creditScores[_user];
        return score == 0 ? baseCreditScore : score;
    }
    
    // Update credit score based on loan performance
    function updateCreditScore(address _user, uint256 _change, bool _increase) internal {
        uint256 currentScore = getCreditScore(_user);
        
        if (_increase) {
            creditScores[_user] = currentScore.add(_change);
            if (creditScores[_user] > 850) {
                creditScores[_user] = 850; // Max score
            }
        } else {
            if (currentScore > _change) {
                creditScores[_user] = currentScore.sub(_change);
            } else {
                creditScores[_user] = 300; // Min score
            }
        }
        
        emit CreditScoreUpdated(_user, creditScores[_user]);
    }
    
    // Update debt cycle indicators (Dalio's principles)
    function updateDebtCycle(uint256 _newDebt, uint256 _newCollateral, uint256 _interestRate) internal {
        currentCycle.totalDebt = currentCycle.totalDebt.add(_newDebt);
        currentCycle.totalCollateral = currentCycle.totalCollateral.add(_newCollateral);
        
        // Calculate weighted average interest rate
        uint256 totalLoans = loanCounter;
        if (totalLoans > 0) {
            currentCycle.averageInterestRate = currentCycle.averageInterestRate
                .mul(totalLoans.sub(1))
                .add(_interestRate)
                .div(totalLoans);
        }
        
        // Calculate credit growth rate (simplified)
        if (currentCycle.timestamp > 0) {
            uint256 timeDiff = block.timestamp.sub(currentCycle.timestamp);
            if (timeDiff > 0) {
                currentCycle.creditGrowthRate = _newDebt.mul(10000).div(timeDiff);
            }
        }
        
        currentCycle.timestamp = block.timestamp;
    }
    
    // Liquidate undercollateralized loan
    function liquidateLoan(uint256 _loanId) external nonReentrant {
        Loan storage loan = loans[_loanId];
        require(loan.isActive, "Loan not active");
        
        uint256 totalDue = calculateTotalDue(_loanId);
        uint256 collateralRatio = loan.collateral.mul(100).div(totalDue);
        
        require(collateralRatio < liquidationThreshold, "Loan not liquidatable");
        
        // Mark loan as defaulted
        loan.isActive = false;
        loan.isDefaulted = true;
        
        // Update borrower's credit score
        updateCreditScore(loan.borrower, 100, false);
        
        // Update default rate
        uint256 defaultedLoans = 0;
        for (uint256 i = 1; i <= loanCounter; i++) {
            if (loans[i].isDefaulted) {
                defaultedLoans++;
            }
        }
        currentCycle.defaultRate = defaultedLoans.mul(10000).div(loanCounter);
        
        // Reward liquidator with a portion of collateral
        uint256 liquidatorReward = loan.collateral.mul(10).div(100); // 10% reward
        payable(msg.sender).transfer(liquidatorReward);
        
        emit LoanLiquidated(_loanId, msg.sender);
    }
    
    // Get debt cycle health score (0-100)
    function getDebtCycleHealth() external view returns (uint256) {
        uint256 healthScore = 100;
        
        // Deduct for high default rate
        if (currentCycle.defaultRate > 100) {
            healthScore = healthScore.sub(currentCycle.defaultRate.div(100));
        }
        
        // Deduct for low collateralization
        if (currentCycle.totalDebt > 0) {
            uint256 collateralRatio = currentCycle.totalCollateral.mul(100).div(currentCycle.totalDebt);
            if (collateralRatio < 150) {
                healthScore = healthScore.sub(150 - collateralRatio);
            }
        }
        
        // Deduct for high credit growth
        if (currentCycle.creditGrowthRate > 1000) {
            healthScore = healthScore.sub(10);
        }
        
        return healthScore > 100 ? 100 : healthScore;
    }
    
    // Get active loans for a user
    function getUserLoans(address _user) external view returns (uint256[] memory) {
        return userLoans[_user];
    }
    
    // Get active investments for a user
    function getUserInvestments(address _user) external view returns (uint256[] memory) {
        return userInvestments[_user];
    }
}