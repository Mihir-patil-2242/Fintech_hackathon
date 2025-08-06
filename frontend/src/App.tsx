import { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

// API Configuration  
const API_URL = 'http://localhost:8000';
// --- IMPORTANT: Update these addresses after deployment ---
const CONTRACT_ADDRESS = '0x5FbDB2315678afecb367f032d93F642f64180aa3';
const LOAN_TOKEN_ADDRESS = '0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512';

// Helper function for API calls
const apiCall = async (endpoint, options = {}) => {
    const token = localStorage.getItem('token');
    const headers = {
        'Content-Type': 'application/json',
        ...options.headers
    };

    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${API_URL}${endpoint}`, {
        ...options,
        headers
    });

    if (!response.ok) {
        const errorBody = await response.text();
        console.error(`API call failed: ${response.statusText}`, errorBody);
        throw new Error(`API call failed: ${response.statusText}`);
    }

    return response.json();
};

export default function FinTechPlatform() {
    const [account, setAccount] = useState(null);
    const [balance, setBalance] = useState('0');
    const [creditScore, setCreditScore] = useState(600);
    const [activeTab, setActiveTab] = useState('dashboard');
    const [loading, setLoading] = useState(false);
    const [marketData, setMarketData] = useState({ crypto: {}, stocks: {} });
    const [debtCycle, setDebtCycle] = useState({ indicators: {} });
    const [chatMessages, setChatMessages] = useState([]);
    const [loanForm, setLoanForm] = useState({
        amount: '',
        duration: 30,
        purpose: 'personal',
        income: '',
        employment: 'employed'
    });
    const [investmentForm, setInvestmentForm] = useState({
        amount: '',
        strategy: 'balanced'
    });

    // Mock wallet connection (since we can't use ethers in artifact)
    const connectWallet = async () => {
        try {
            // Simulate wallet connection
            const mockAddress = '0x' + Math.random().toString(16).substr(2, 40);
            setAccount(mockAddress);

            // Mock authentication
            await authenticate(mockAddress);

            // Set mock data
            setBalance((Math.random() * 1000).toFixed(2));
            setCreditScore(Math.floor(Math.random() * 250) + 600);
        } catch (error) {
            console.error('Failed to connect wallet:', error);
        }
    };

    // Authenticate with backend
    const authenticate = async (address) => {
        try {
            // Get nonce
            const nonceData = await apiCall(`/auth/nonce?wallet_address=${address}`);

            // Mock signature (in real app, would use MetaMask)
            const mockSignature = '0x' + Math.random().toString(16).substr(2, 130);

            // Login
            const loginData = await apiCall('/auth/login', {
                method: 'POST',
                body: JSON.stringify({
                    wallet_address: address,
                    signature: mockSignature,
                    message: nonceData.message
                })
            });

            // Store token
            localStorage.setItem('token', loginData.access_token);
            setCreditScore(loginData.credit_score || 600);
        } catch (error) {
            console.error('Authentication failed:', error);
        }
    };

    // Apply for loan
    const applyForLoan = async () => {
        setLoading(true);
        try {
            const response = await apiCall('/loans/apply', {
                method: 'POST',
                body: JSON.stringify(loanForm)
            });

            alert(`Loan application submitted! Required collateral: ${response.loan_application.collateral_required} ETH`);
        } catch (error) {
            console.error('Loan application failed:', error);
            alert('Failed to submit loan application');
        }
        setLoading(false);
    };

    // Create investment
    const createInvestment = async () => {
        setLoading(true);
        try {
            const response = await apiCall('/investments/create', {
                method: 'POST',
                body: JSON.stringify(investmentForm)
            });

            alert(`Investment plan created! Expected APY: ${response.investment_plan.expected_apy}%`);
        } catch (error) {
            console.error('Investment creation failed:', error);
            alert('Failed to create investment');
        }
        setLoading(false);
    };

    // Fetch market data
    const fetchMarketData = async () => {
        try {
            const data = await apiCall('/market/data');
            setMarketData(data);
        } catch (error) {
            console.error('Failed to fetch market data:', error);
            // Set mock data
            setMarketData({
                crypto: {
                    BTC: { price: 45000, change_24h: 2.5, volume: 25000000 },
                    ETH: { price: 3200, change_24h: -1.2, volume: 15000000 }
                },
                stocks: {
                    AAPL: { price: 175, change: 1.8, volume: 50000000 },
                    GOOGL: { price: 140, change: -0.5, volume: 30000000 }
                }
            });
        }
    };

    // Fetch debt cycle
    const fetchDebtCycle = async () => {
        try {
            const data = await apiCall('/market/debt-cycle');
            setDebtCycle(data);
        } catch (error) {
            console.error('Failed to fetch debt cycle:', error);
            // Set mock data
            setDebtCycle({
                indicators: {
                    health_score: 75,
                    cycle_phase: 'expansion',
                    total_debt: 1000000,
                    total_collateral: 1500000,
                    default_rate: 2.5
                }
            });
        }
    };

    // Chat with AI
    const sendChatMessage = async (message) => {
        try {
            const response = await apiCall('/chat', {
                method: 'POST',
                body: JSON.stringify({ message })
            });

            setChatMessages([
                ...chatMessages,
                { user: message, ai: response.response }
            ]);
        } catch (error) {
            console.error('Chat failed:', error);
            // Add mock response
            setChatMessages([
                ...chatMessages,
                { user: message, ai: 'I can help you with loans, investments, and financial planning. How can I assist you today?' }
            ]);
        }
    };

    useEffect(() => {
        if (localStorage.getItem('token')) {
            fetchMarketData();
            fetchDebtCycle();
        }

        const interval = setInterval(() => {
            if (localStorage.getItem('token')) {
                fetchMarketData();
                fetchDebtCycle();
            }
        }, 30000);

        return () => clearInterval(interval);
    }, []);

    // Chart data
    const debtCycleChartData = [
        { name: 'Total Debt', value: debtCycle.indicators?.total_debt || 1000000 },
        { name: 'Collateral', value: debtCycle.indicators?.total_collateral || 1500000 }
    ];

    const marketChartData = Object.entries(marketData.crypto || {}).map(([symbol, data]) => ({
        name: symbol,
        price: data.price || 0,
        change: data.change_24h || 0
    }));

    const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff6b6b'];

    // Dashboard Component
    const Dashboard = () => (
        <div className="animate-fadeIn">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                {/* Portfolio Card */}
                <div className="bg-gradient-to-br from-purple-600 to-blue-600 rounded-2xl p-6 text-white shadow-xl transform hover:scale-105 transition-transform">
                    <h3 className="text-xl font-bold mb-4">Portfolio Value</h3>
                    <div className="text-3xl font-bold mb-2">{balance} FLT</div>
                    <div className="text-sm opacity-80">≈ ${(parseFloat(balance) * 10).toFixed(2)} USD</div>
                    <div className="mt-4 space-y-2">
                        <div className="flex justify-between">
                            <span>Active Loans</span>
                            <span>2</span>
                        </div>
                        <div className="flex justify-between">
                            <span>Investments</span>
                            <span>3</span>
                        </div>
                    </div>
                </div>

                {/* Credit Score Card */}
                <div className="bg-gradient-to-br from-green-600 to-teal-600 rounded-2xl p-6 text-white shadow-xl transform hover:scale-105 transition-transform">
                    <h3 className="text-xl font-bold mb-4">Credit Score</h3>
                    <div className="relative">
                        <svg className="w-32 h-32 mx-auto">
                            <circle
                                cx="64"
                                cy="64"
                                r="56"
                                stroke="rgba(255,255,255,0.2)"
                                strokeWidth="12"
                                fill="none"
                            />
                            <circle
                                cx="64"
                                cy="64"
                                r="56"
                                stroke="white"
                                strokeWidth="12"
                                fill="none"
                                strokeDasharray={`${(creditScore / 850) * 352} 352`}
                                strokeLinecap="round"
                                transform="rotate(-90 64 64)"
                            />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center">
                            <div className="text-center">
                                <div className="text-2xl font-bold">{creditScore}</div>
                                <div className="text-xs opacity-80">
                                    {creditScore > 750 ? 'Excellent' : creditScore > 650 ? 'Good' : 'Fair'}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Debt Cycle Card */}
                <div className="bg-gradient-to-br from-orange-600 to-red-600 rounded-2xl p-6 text-white shadow-xl transform hover:scale-105 transition-transform">
                    <h3 className="text-xl font-bold mb-4">Debt Cycle Health</h3>
                    <div className="text-3xl font-bold mb-2">
                        {debtCycle.indicators?.health_score || 75}/100
                    </div>
                    <div className="text-sm opacity-80 capitalize">
                        Phase: {debtCycle.indicators?.cycle_phase || 'Expansion'}
                    </div>
                    <div className="mt-4">
                        <div className="bg-white/20 rounded-full h-2 mb-2">
                            <div
                                className="bg-white rounded-full h-2 transition-all duration-500"
                                style={{ width: `${debtCycle.indicators?.health_score || 75}%` }}
                            />
                        </div>
                        <div className="text-xs opacity-80">
                            {(debtCycle.indicators?.health_score || 75) > 60 ? 'Healthy' : 'Caution Advised'}
                        </div>
                    </div>
                </div>
            </div>

lg font-semibold mb-2">Investments</h3>
                        <div className="text-3xl font-bold mb-1">{portfolio?.investments?.length || 0}</div>
                        <div className="text-sm opacity-80">Active positions</div>
                    </div>
                </div>

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="bg-white rounded-2xl p-6 shadow-lg">
                        <h3 className="text-xl font-bold mb-4">Market Overview</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={marketChartData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="name" />
                                <YAxis />
                                <Tooltip formatter={(value) => [`${value.toLocaleString()}`, 'Price']} />
                                <Bar dataKey="price" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="bg-white rounded-2xl p-6 shadow-lg">
                        <h3 className="text-xl font-bold mb-4">Debt Cycle Health</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie
                                    data={debtCycleChartData}
                                    cx="50%"
                                    cy="50%"
                                    outerRadius={80}
                                    dataKey="value"
                                    label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                                >
                                    {debtCycleChartData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip formatter={(value) => `${value.toLocaleString()}`} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Market Data Table */}
                <div className="bg-white rounded-2xl p-6 shadow-lg">
                    <h3 className="text-xl font-bold mb-4">Live Market Data</h3>
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b">
                                    <th className="text-left py-3">Asset</th>
                                    <th className="text-right py-3">Price</th>
                                    <th className="text-right py-3">24h Change</th>
                                    <th className="text-right py-3">Volume</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries({ ...marketData.crypto, ...marketData.stocks }).map(([symbol, data]: [string, any]) => (
                                    <tr key={symbol} className="border-b hover:bg-gray-50 transition">
                                        <td className="py-3 font-medium">{symbol}</td>
                                        <td className="text-right font-mono">${data.price?.toFixed(2) || '0.00'}</td>
                                        <td className={`text-right font-mono ${(data.change_24h || data.change || 0) > 0 ? 'text-green-500' : 'text-red-500'}`}>
                                            {(data.change_24h || data.change || 0) > 0 ? '+' : ''}{(data.change_24h || data.change || 0).toFixed(2)}%
                                        </td>
                                        <td className="text-right font-mono">${((data.volume || 0) / 1000000).toFixed(1)}M</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        );
    };

    const LoanApplication = () => (
        <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-2xl p-8 shadow-lg">
                <h2 className="text-3xl font-bold mb-2">Apply for a DeFi Loan</h2>
                <p className="text-gray-600 mb-8">Get instant funding with crypto collateral</p>

                <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium mb-2 text-gray-700">Loan Amount (FLT)</label>
                            <input
                                type="number"
                                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                                value={loanForm.amount}
                                onChange={(e) => setLoanForm({ ...loanForm, amount: e.target.value })}
                                placeholder="10,000"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-2 text-gray-700">Duration (days)</label>
                            <select
                                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                                value={loanForm.duration}
                                onChange={(e) => setLoanForm({ ...loanForm, duration: parseInt(e.target.value) })}
                            >
                                <option value={30}>30 days</option>
                                <option value={90}>90 days</option>
                                <option value={180}>180 days</option>
                                <option value={365}>1 year</option>
                            </select>
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-2 text-gray-700">Annual Income</label>
                        <input
                            type="number"
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                            value={loanForm.income}
                            onChange={(e) => setLoanForm({ ...loanForm, income: e.target.value })}
                            placeholder="75,000"
                        />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium mb-2 text-gray-700">Purpose</label>
                            <select
                                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                                value={loanForm.purpose}
                                onChange={(e) => setLoanForm({ ...loanForm, purpose: e.target.value })}
                            >
                                <option value="personal">Personal</option>
                                <option value="business">Business</option>
                                <option value="education">Education</option>
                                <option value="medical">Medical</option>
                                <option value="investment">Investment</option>
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-2 text-gray-700">Employment</label>
                            <select
                                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                                value={loanForm.employment}
                                onChange={(e) => setLoanForm({ ...loanForm, employment: e.target.value })}
                            >
                                <option value="employed">Employed</option>
                                <option value="self-employed">Self Employed</option>
                                <option value="freelancer">Freelancer</option>
                                <option value="student">Student</option>
                                <option value="retired">Retired</option>
                            </select>
                        </div>
                    </div>

                    {loanForm.amount && loanForm.income && (
                        <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
                            <h4 className="font-semibold text-blue-900 mb-4">💡 Loan Preview</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                                <div className="flex justify-between">
                                    <span>Interest Rate:</span>
                                    <span className="font-semibold">8.5% APR</span>
                                </div>
                                <div className="flex justify-between">
                                    <span>Collateral Required:</span>
                                    <span className="font-semibold">{(parseFloat(loanForm.amount) * 1.5).toFixed(0)} FLT</span>
                                </div>
                                <div className="flex justify-between">
                                    <span>Monthly Payment:</span>
                                    <span className="font-semibold">{(parseFloat(loanForm.amount) * 1.085 / (loanForm.duration/30)).toFixed(0)} FLT</span>
                                </div>
                                <div className="flex justify-between">
                                    <span>Total Repayment:</span>
                                    <span className="font-semibold">{(parseFloat(loanForm.amount) * 1.085).toFixed(0)} FLT</span>
                                </div>
                            </div>
                        </div>
                    )}

                    <button
                        onClick={applyForLoan}
                        disabled={loading || !account || !loanForm.amount || !loanForm.income}
                        className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 rounded-xl font-semibold text-lg hover:opacity-90 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
                    >
                        {loading ? '⏳ Processing...' : '🚀 Submit Application'}
                    </button>
                </div>
            </div>
        );

    // Investment Component
    const InvestmentPools = () => (
        <div className="max-w-4xl mx-auto animate-fadeIn">
            <div className="bg-white rounded-2xl p-8 shadow-lg mb-6">
                <h2 className="text-2xl font-bold mb-6">Investment Pools</h2>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="border rounded-lg p-4 hover:shadow-md transition">
                        <h3 className="font-semibold mb-2">Conservative Pool</h3>
                        <div className="text-2xl font-bold text-green-600 mb-2">5-7% APY</div>
                        <p className="text-sm text-gray-600">Low risk, stable returns</p>
                        <ul className="text-xs text-gray-500 mt-2">
                            <li>• Government bonds</li>
                            <li>• Stablecoins</li>
                            <li>• Blue-chip stocks</li>
                        </ul>
                    </div>

                    <div className="border-2 border-blue-500 rounded-lg p-4 hover:shadow-md transition relative">
                        <div className="absolute top-2 right-2 bg-blue-500 text-white text-xs px-2 py-1 rounded-full">Recommended</div>
                        <h3 className="font-semibold mb-2">Balanced Pool</h3>
                        <div className="text-2xl font-bold text-blue-600 mb-2">8-10% APY</div>
                        <p className="text-sm text-gray-600">Moderate risk, good returns</p>
                        <ul className="text-xs text-gray-500 mt-2">
                            <li>• Mixed portfolio</li>
                            <li>• ETH staking</li>
                            <li>• Index funds</li>
                        </ul>
                    </div>

                    <div className="border rounded-lg p-4 hover:shadow-md transition">
                        <h3 className="font-semibold mb-2">Aggressive Pool</h3>
                        <div className="text-2xl font-bold text-red-600 mb-2">12-15% APY</div>
                        <p className="text-sm text-gray-600">High risk, high returns</p>
                        <ul className="text-xs text-gray-500 mt-2">
                            <li>• DeFi yield farming</li>
                            <li>• Growth stocks</li>
                            <li>• Crypto trading</li>
                        </ul>
                    </div>
                </div>

                <div className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium mb-2">Investment Amount (FLT)</label>
                        <input
                            type="number"
                            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                            value={investmentForm.amount}
                            onChange={(e) => setInvestmentForm({ ...investmentForm, amount: e.target.value })}
                            placeholder="Enter amount to invest"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-2">Strategy</label>
                        <select
                            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                            value={investmentForm.strategy}
                            onChange={(e) => setInvestmentForm({ ...investmentForm, strategy: e.target.value })}
                        >
                            <option value="conservative">Conservative (5-7% APY)</option>
                            <option value="balanced">Balanced (8-10% APY)</option>
                            <option value="aggressive">Aggressive (12-15% APY)</option>
                        </select>
                    </div>

                    {investmentForm.amount && (
                        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                            <h4 className="font-semibold text-green-900 mb-2">Investment Preview</h4>
                            <div className="space-y-1 text-sm">
                                <div className="flex justify-between">
                                    <span>Expected Annual Return:</span>
                                    <span className="font-medium">
                                        {investmentForm.strategy === 'conservative' ? '6%' :
                                            investmentForm.strategy === 'balanced' ? '9%' : '13.5%'}
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span>Estimated Yearly Profit:</span>
                                    <span className="font-medium">
                                        {(parseFloat(investmentForm.amount) *
                                            (investmentForm.strategy === 'conservative' ? 0.06 :
                                                investmentForm.strategy === 'balanced' ? 0.09 : 0.135)).toFixed(2)} FLT
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}

                    <button
                        onClick={createInvestment}
                        disabled={loading || !account || !investmentForm.amount}
                        className="w-full bg-gradient-to-r from-green-600 to-teal-600 text-white py-3 rounded-lg font-semibold hover:opacity-90 transition disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? 'Processing...' : 'Start Investing'}
                    </button>
                </div>
            </div>
        </div>
    );

    // AI Chat Component
    const AIChat = () => {
        const [input, setInput] = useState('');

        return (
            <div className="max-w-4xl mx-auto">
                <div className="bg-white rounded-2xl shadow-lg h-[600px] flex flex-col">
                    <div className="p-6 border-b bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-t-2xl">
                        <h2 className="text-2xl font-bold">AI Financial Advisor</h2>
                        <p className="text-sm opacity-90">Powered by Gemini Pro - Get personalized financial advice</p>
                    </div>

                    <div className="flex-1 overflow-y-auto p-6 space-y-4">
                        {chatMessages.length === 0 && (
                            <div className="text-center text-gray-500 py-12">
                                <div className="text-6xl mb-4">🤖</div>
                                <h3 className="text-xl font-semibold mb-2">AI Financial Advisor</h3>
                                <p className="mb-4">Ask me about loans, investments, market analysis, or financial planning.</p>
                                <div className="flex flex-wrap justify-center gap-2">
                                    <button
                                        onClick={() => sendChatMessage("What's my credit score and how can I improve it?")}
                                        className="text-xs px-3 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition"
                                    >
                                        Credit Score Help
                                    </button>
                                    <button
                                        onClick={() => sendChatMessage("Give me personalized investment advice")}
                                        className="text-xs px-3 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition"
                                    >
                                        Investment Strategy
                                    </button>
                                    <button
                                        onClick={() => sendChatMessage("Analyze current market conditions")}
                                        className="text-xs px-3 py-2 bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition"
                                    >
                                        Market Analysis
                                    </button>
                                </div>
                            </div>
                        )}

                        {chatMessages.map((msg, i) => (
                            <div key={i} className="space-y-3">
                                <div className="flex justify-end">
                                    <div className="bg-blue-600 text-white rounded-2xl px-4 py-3 max-w-xs shadow-md">
                                        {msg.user}
                                    </div>
                                </div>
                                <div className="flex justify-start">
                                    <div className="bg-gray-100 rounded-2xl px-4 py-3 max-w-md shadow-md">
                                        {msg.ai}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="p-6 border-t bg-gray-50 rounded-b-2xl">
                        <div className="flex gap-3">
                            <input
                                type="text"
                                className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                                placeholder="Ask about loans, investments, market analysis..."
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyPress={(e) => {
                                    if (e.key === 'Enter' && input.trim()) {
                                        sendChatMessage(input);
                                        setInput('');
                                    }
                                }}
                            />
                            <button
                                onClick={() => {
                                    if (input.trim()) {
                                        sendChatMessage(input);
                                        setInput('');
                                    }
                                }}
                                disabled={!input.trim()}
                                className="px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed font-semibold"
                            >
                                Send
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
            {/* Header */}
            <header className="bg-white/95 backdrop-blur-sm shadow-lg sticky top-0 z-50">
                <div className="container mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                            <h1 
                                className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent cursor-pointer"
                                onClick={() => setActiveTab('home')}
                            >
                                FinTech Platform v2
                            </h1>
                            <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full font-semibold">DeFi</span>
                            <div className={`w-2 h-2 rounded-full ${
                                apiStatus === 'connected' ? 'bg-green-400' : 'bg-red-400'
                            }`}></div>
                        </div>

                        <nav className="hidden md:flex space-x-1">
                            {[
                                { id: 'home', label: 'Home', icon: '🏠' },
                                { id: 'dashboard', label: 'Dashboard', icon: '📈' },
                                { id: 'loans', label: 'Loans', icon: '🏦' },
                                { id: 'investments', label: 'Invest', icon: '💰' },
                                { id: 'chat', label: 'AI Advisor', icon: '🤖' }
                            ].map(tab => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`px-4 py-2 rounded-lg transition-all ${
                                        activeTab === tab.id 
                                            ? 'bg-blue-600 text-white shadow-md' 
                                            : 'hover:bg-gray-100 text-gray-700'
                                    }`}
                                >
                                    <span className="mr-1">{tab.icon}</span>
                                    {tab.label}
                                </button>
                            ))}
                        </nav>

                        <div>
                            {account ? (
                                <div className="flex items-center space-x-4">
                                    <div className="text-sm text-right">
                                        <div className="text-gray-500">Connected</div>
                                        <div className="font-mono text-xs font-semibold">{account.slice(0, 6)}...{account.slice(-4)}</div>
                                    </div>
                                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                                </div>
                            ) : (
                                <button
                                    onClick={connectWallet}
                                    disabled={loading}
                                    className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:opacity-90 transition font-semibold disabled:opacity-50"
                                >
                                    {loading ? 'Connecting...' : 'Connect Wallet'}
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="container mx-auto px-6 py-8">
                {activeTab === 'home' && <Homepage />}
                {activeTab === 'dashboard' && (account ? <Dashboard /> : <Homepage />)}
                {activeTab === 'loans' && (account ? <LoanApplication /> : <Homepage />)}
                {activeTab === 'investments' && (account ? <InvestmentPools /> : <Homepage />)}
                {activeTab === 'chat' && (account ? <AIChat /> : <Homepage />)}
            </main>

            {/* Footer */}
            <footer className="bg-white/95 backdrop-blur-sm mt-12 py-8 border-t">
                <div className="container mx-auto px-6">
                    <div className="flex flex-col md:flex-row justify-between items-center">
                        <div className="mb-4 md:mb-0">
                            <h3 className="font-bold text-gray-800">FinTech Platform v2</h3>
                            <p className="text-sm text-gray-600">Powered by Blockchain & AI</p>
                        </div>
                        <div className="flex space-x-6 text-sm text-gray-600">
                            <a href="#" className="hover:text-blue-600 transition">Documentation</a>
                            <a href="#" className="hover:text-blue-600 transition">API</a>
                            <a href={`${API_URL}/docs`} target="_blank" className="hover:text-blue-600 transition">Backend Docs</a>
                            <a href="#" className="hover:text-blue-600 transition">Support</a>
                        </div>
                    </div>
                    <div className="mt-4 text-center text-xs text-gray-500">
                        Built with React, FastAPI, and Web3 • {new Date().getFullYear()}
                    </div>
                </div>
            </footer>
        </div>
    );
}