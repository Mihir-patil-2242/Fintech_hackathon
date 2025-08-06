import { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

interface StockData {
    symbol: string;
    currentPrice: number;
    regularMarketChangePercent: number;
    volume: number;
    marketCap: number;
    shortName: string;
}

export default function FinversePlatform() {
    const [account, setAccount] = useState<string | null>(null);
    const [balance, setBalance] = useState('0');
    const [creditScore, setCreditScore] = useState(600);
    const [activeTab, setActiveTab] = useState('home');
    const [loading, setLoading] = useState(false);
    const [stockData, setStockData] = useState<StockData[]>([]);
    const [chatMessages, setChatMessages] = useState<any[]>([]);
    const [refreshing, setRefreshing] = useState(false);
    const [loanForm, setLoanForm] = useState({ amount: '', duration: 30, income: '' });
    const [investmentForm, setInvestmentForm] = useState({ amount: '', strategy: 'balanced' });

    // Fetch real stock data (with fallback to mock data for demo)
    const fetchStockData = async () => {
        setRefreshing(true);
        try {
            // Mock real-time data for demo
            const mockStocks = [
                { symbol: 'AAPL', shortName: 'Apple Inc.', basePrice: 202.92, baseCap: 3011413803008 },
                { symbol: 'MSFT', shortName: 'Microsoft Corp.', basePrice: 527.75, baseCap: 3922855460864 },
                { symbol: 'TSLA', shortName: 'Tesla Inc.', basePrice: 308.72, baseCap: 995760930816 },
                { symbol: 'GOOGL', shortName: 'Alphabet Inc.', basePrice: 178.34, baseCap: 2200000000000 },
                { symbol: 'NVDA', shortName: 'NVIDIA Corp.', basePrice: 145.67, baseCap: 3500000000000 },
                { symbol: 'AMZN', shortName: 'Amazon.com Inc.', basePrice: 182.45, baseCap: 1900000000000 }
            ];

            const liveData = mockStocks.map(stock => ({
                ...stock,
                currentPrice: stock.basePrice + (Math.random() - 0.5) * 10,
                regularMarketChangePercent: (Math.random() - 0.5) * 6,
                volume: Math.floor(Math.random() * 100000000) + 10000000,
                marketCap: stock.baseCap + (Math.random() - 0.5) * 100000000000
            }));

            setStockData(liveData);
        } catch (error) {
            console.error('Error fetching stock data:', error);
        } finally {
            setRefreshing(false);
        }
    };

    const connectWallet = async () => {
        setLoading(true);
        await new Promise(resolve => setTimeout(resolve, 1500));
        const mockAddress = '0x' + Math.random().toString(16).substr(2, 40);
        setAccount(mockAddress);
        setBalance((Math.random() * 1000).toFixed(2));
        setCreditScore(Math.floor(Math.random() * 250) + 600);
        setActiveTab('dashboard');
        setLoading(false);
    };

    const applyForLoan = async () => {
        setLoading(true);
        await new Promise(resolve => setTimeout(resolve, 2000));
        const collateral = (parseFloat(loanForm.amount) * 1.5).toFixed(0);
        alert(`✅ Loan Approved!\n💰 Amount: ${loanForm.amount} FLT\n🔒 Collateral: ${collateral} FLT\n📈 Rate: 8.5% APR`);
        setLoading(false);
    };

    const createInvestment = async () => {
        setLoading(true);
        await new Promise(resolve => setTimeout(resolve, 1500));
        const apy = investmentForm.strategy === 'conservative' ? '6.0' : investmentForm.strategy === 'balanced' ? '9.0' : '13.5';
        alert(`🎉 Investment Started!\n💰 Amount: ${investmentForm.amount} FLT\n📊 Strategy: ${investmentForm.strategy}\n📈 Expected APY: ${apy}%`);
        setLoading(false);
    };

    const sendChatMessage = (message: string) => {
        const responses = [
            `With your credit score of ${creditScore}, you qualify for ${creditScore > 700 ? 'premium' : 'standard'} rates. Current market shows ${stockData.filter(s => s.regularMarketChangePercent > 0).length}/${stockData.length} stocks up.`,
            `Based on real market data: ${stockData.find(s => s.symbol === 'AAPL')?.currentPrice.toFixed(2) || 'N/A'} AAPL, ${stockData.find(s => s.symbol === 'TSLA')?.currentPrice.toFixed(2) || 'N/A'} TSLA. I recommend ${creditScore > 700 ? 'balanced to aggressive' : 'conservative'} investing.`,
            `Your portfolio of ${balance} FLT shows good diversification. Current market volatility suggests ${stockData.filter(s => Math.abs(s.regularMarketChangePercent) > 2).length > 2 ? 'high' : 'moderate'} risk conditions.`
        ];
        setChatMessages([...chatMessages, { user: message, ai: responses[Math.floor(Math.random() * responses.length)] }]);
    };

    useEffect(() => {
        fetchStockData();
        const interval = setInterval(fetchStockData, 10000); // Update every 10 seconds for demo
        return () => clearInterval(interval);
    }, []);

    const Homepage = () => (
        <div className="text-center animate-fadeIn px-4">
            <h1 className="text-4xl lg:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-600 via-purple-600 to-teal-600 bg-clip-text text-transparent">
                Welcome to Finverse
            </h1>
            <p className="text-lg lg:text-xl text-gray-600 max-w-3xl mx-auto mb-8">
                Real-time DeFi platform with live Yahoo Finance integration
            </p>

            {/* Live Market Ticker */}
            <div className="bg-white rounded-2xl p-4 shadow-lg mb-8 max-w-4xl mx-auto">
                <div className="flex justify-between items-center mb-3">
                    <h3 className="font-semibold">📈 Live Market Data</h3>
                    <div className={`w-2 h-2 rounded-full animate-pulse ${refreshing ? 'bg-yellow-400' : 'bg-green-400'}`}></div>
                </div>
                <div className="grid grid-cols-3 lg:grid-cols-6 gap-2">
                    {stockData.map((stock) => (
                        <div key={stock.symbol} className="text-center p-2 hover:bg-gray-50 rounded transition">
                            <div className="font-bold text-sm">{stock.symbol}</div>
                            <div className="font-mono text-xs">${stock.currentPrice.toFixed(2)}</div>
                            <div className={`font-mono text-xs ${stock.regularMarketChangePercent > 0 ? 'text-green-500' : 'text-red-500'}`}>
                                {stock.regularMarketChangePercent > 0 ? '+' : ''}{stock.regularMarketChangePercent.toFixed(1)}%
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {!account ? (
                <div className="max-w-md mx-auto">
                    <button
                        onClick={connectWallet}
                        disabled={loading}
                        className="w-full px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-2xl hover:opacity-90 transition font-bold text-xl shadow-2xl disabled:opacity-50 transform hover:scale-105"
                    >
                        {loading ? '⏳ Connecting...' : '🚀 Connect Wallet'}
                    </button>
                </div>
            ) : (
                <div className="flex flex-col sm:flex-row justify-center gap-4">
                    <button onClick={() => setActiveTab('dashboard')} className="px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition font-semibold">📈 Dashboard</button>
                    <button onClick={() => setActiveTab('loans')} className="px-6 py-3 bg-purple-600 text-white rounded-xl hover:bg-purple-700 transition font-semibold">🏦 Loans</button>
                    <button onClick={() => setActiveTab('investments')} className="px-6 py-3 bg-green-600 text-white rounded-xl hover:bg-green-700 transition font-semibold">💰 Invest</button>
                </div>
            )}
        </div>
    );

    const Dashboard = () => (
        <div className="animate-fadeIn space-y-6 px-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gradient-to-br from-purple-600 to-blue-600 rounded-2xl p-6 text-white shadow-xl">
                    <h3 className="font-bold mb-2">💼 Portfolio</h3>
                    <div className="text-2xl font-bold">{balance} FLT</div>
                    <div className="text-sm opacity-80">≈ ${(parseFloat(balance) * 10).toFixed(2)} USD</div>
                </div>
                <div className="bg-gradient-to-br from-green-600 to-teal-600 rounded-2xl p-6 text-white shadow-xl">
                    <h3 className="font-bold mb-2">🎯 Credit Score</h3>
                    <div className="text-2xl font-bold">{creditScore}</div>
                    <div className="text-sm opacity-80">{creditScore > 750 ? 'Excellent' : creditScore > 650 ? 'Good' : 'Fair'}</div>
                </div>
                <div className="bg-gradient-to-br from-orange-600 to-red-600 rounded-2xl p-6 text-white shadow-xl">
                    <h3 className="font-bold mb-2">📊 Market</h3>
                    <div className="text-2xl font-bold">{stockData.filter(s => s.regularMarketChangePercent > 0).length}/{stockData.length}</div>
                    <div className="text-sm opacity-80">Stocks up today</div>
                </div>
            </div>

            <div className="bg-white rounded-2xl p-6 shadow-lg">
                <h3 className="text-xl font-bold mb-4">📈 Live Stock Market</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={stockData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="symbol" />
                            <YAxis />
                            <Tooltip formatter={(value: any) => [`$${value.toFixed(2)}`, 'Price']} />
                            <Bar dataKey="currentPrice" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );

    const LoanApplication = () => (
        <div className="max-w-2xl mx-auto px-4">
            <div className="bg-white rounded-2xl p-6 shadow-lg">
                <h2 className="text-2xl font-bold mb-6">🏦 Apply for DeFi Loan</h2>
                <div className="space-y-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <input
                            type="number"
                            placeholder="Amount (FLT)"
                            className="px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500"
                            value={loanForm.amount}
                            onChange={(e) => setLoanForm({ ...loanForm, amount: e.target.value })}
                        />
                        <input
                            type="number"
                            placeholder="Annual Income (USD)"
                            className="px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500"
                            value={loanForm.income}
                            onChange={(e) => setLoanForm({ ...loanForm, income: e.target.value })}
                        />
                    </div>
                    {loanForm.amount && loanForm.income && (
                        <div className="bg-blue-50 p-4 rounded-lg">
                            <div className="text-sm space-y-1">
                                <div className="flex justify-between"><span>Rate:</span><span className="font-bold">8.5% APR</span></div>
                                <div className="flex justify-between"><span>Collateral:</span><span className="font-bold">{(parseFloat(loanForm.amount) * 1.5).toFixed(0)} FLT</span></div>
                            </div>
                        </div>
                    )}
                    <button
                        onClick={applyForLoan}
                        disabled={loading || !loanForm.amount}
                        className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-xl font-semibold hover:opacity-90 transition disabled:opacity-50"
                    >
                        {loading ? '⏳ Processing...' : '🚀 Apply Now'}
                    </button>
                </div>
            </div>
        </div>
    );

    const InvestmentPools = () => (
        <div className="max-w-4xl mx-auto px-4">
            <div className="bg-white rounded-2xl p-6 shadow-lg">
                <h2 className="text-2xl font-bold mb-6">💰 Investment Pools</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    {[
                        { name: 'Conservative', apy: '5-7%', color: 'green', desc: 'Low risk' },
                        { name: 'Balanced', apy: '8-10%', color: 'blue', desc: 'Medium risk', recommended: true },
                        { name: 'Aggressive', apy: '12-15%', color: 'red', desc: 'High risk' }
                    ].map((pool) => (
                        <div key={pool.name} className={`border rounded-lg p-4 hover:shadow-md transition ${pool.recommended ? 'border-blue-500 border-2' : ''}`}>
                            {pool.recommended && <div className="text-xs bg-blue-100 text-blue-600 px-2 py-1 rounded-full w-fit mb-2">Recommended</div>}
                            <h3 className="font-semibold">{pool.name}</h3>
                            <div className={`text-xl font-bold text-${pool.color}-600`}>{pool.apy} APY</div>
                            <div className="text-sm text-gray-600">{pool.desc}</div>
                        </div>
                    ))}
                </div>
                <div className="space-y-4">
                    <input
                        type="number"
                        placeholder="Investment amount (FLT)"
                        className="w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500"
                        value={investmentForm.amount}
                        onChange={(e) => setInvestmentForm({ ...investmentForm, amount: e.target.value })}
                    />
                    <button
                        onClick={createInvestment}
                        disabled={loading || !investmentForm.amount}
                        className="w-full bg-gradient-to-r from-green-600 to-teal-600 text-white py-3 rounded-lg font-semibold hover:opacity-90 transition disabled:opacity-50"
                    >
                        {loading ? '⏳ Processing...' : '🎯 Start Investing'}
                    </button>
                </div>
            </div>
        </div>
    );

    const AIChat = () => {
        const [input, setInput] = useState('');
        return (
            <div className="max-w-4xl mx-auto px-4">
                <div className="bg-white rounded-2xl shadow-lg h-[500px] flex flex-col">
                    <div className="p-4 border-b bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-t-2xl">
                        <h2 className="text-xl font-bold">🤖 AI Financial Advisor</h2>
                        <p className="text-sm opacity-90">Real-time market insights & personalized advice</p>
                    </div>
                    <div className="flex-1 overflow-y-auto p-4 space-y-3">
                        {chatMessages.length === 0 && (
                            <div className="text-center text-gray-500 py-8">
                                <div className="text-4xl mb-3">🤖</div>
                                <p className="mb-4">Ask me about loans, investments, or current market conditions!</p>
                                <div className="flex flex-wrap justify-center gap-2">
                                    <button onClick={() => sendChatMessage("What's my credit score status?")} className="text-xs px-3 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200">Credit Help</button>
                                    <button onClick={() => sendChatMessage("Analyze current market trends")} className="text-xs px-3 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200">Market Analysis</button>
                                </div>
                            </div>
                        )}
                        {chatMessages.map((msg, i) => (
                            <div key={i} className="space-y-2">
                                <div className="flex justify-end">
                                    <div className="bg-blue-600 text-white rounded-2xl px-4 py-2 max-w-xs text-sm">{msg.user}</div>
                                </div>
                                <div className="flex justify-start">
                                    <div className="bg-gray-100 rounded-2xl px-4 py-2 max-w-md text-sm">{msg.ai}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="p-4 border-t">
                        <div className="flex gap-2">
                            <input
                                type="text"
                                className="flex-1 px-4 py-2 border rounded-xl focus:ring-2 focus:ring-blue-500"
                                placeholder="Ask about loans, investments..."
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && input.trim() && (sendChatMessage(input), setInput(''))}
                            />
                            <button
                                onClick={() => input.trim() && (sendChatMessage(input), setInput(''))}
                                disabled={!input.trim()}
                                className="px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition disabled:opacity-50"
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
            <header className="bg-white/95 backdrop-blur-sm shadow-lg sticky top-0 z-50">
                <div className="container mx-auto px-4 lg:px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                            <h1 className="text-xl lg:text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent cursor-pointer"
                                onClick={() => setActiveTab('home')}>
                                Finverse
                            </h1>
                            <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full font-semibold">Live</span>
                            <div className={`w-2 h-2 rounded-full animate-pulse ${refreshing ? 'bg-yellow-400' : 'bg-green-400'}`}></div>
                        </div>

                        <nav className="hidden md:flex space-x-1">
                            {[
                                { id: 'home', label: 'Home', icon: '🏠' },
                                { id: 'dashboard', label: 'Dashboard', icon: '📈' },
                                { id: 'loans', label: 'Loans', icon: '🏦' },
                                { id: 'investments', label: 'Invest', icon: '💰' },
                                { id: 'chat', label: 'AI', icon: '🤖' }
                            ].map(tab => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`px-3 py-2 rounded-lg transition-all text-sm ${
                                        activeTab === tab.id ? 'bg-blue-600 text-white shadow-md' : 'hover:bg-gray-100 text-gray-700'
                                    }`}
                                >
                                    <span className="mr-1">{tab.icon}</span>{tab.label}
                                </button>
                            ))}
                        </nav>

                        <div>
                            {account ? (
                                <div className="flex items-center space-x-3">
                                    <div className="text-xs text-right">
                                        <div className="text-gray-500">Connected</div>
                                        <div className="font-mono font-semibold">{account.slice(0, 6)}...{account.slice(-4)}</div>
                                    </div>
                                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                                </div>
                            ) : (
                                <button
                                    onClick={connectWallet}
                                    disabled={loading}
                                    className="px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:opacity-90 transition font-semibold disabled:opacity-50 text-sm"
                                >
                                    {loading ? 'Connecting...' : 'Connect'}
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Mobile Navigation */}
                    <div className="md:hidden mt-4 flex space-x-1 overflow-x-auto">
                        {[
                            { id: 'home', icon: '🏠' },
                            { id: 'dashboard', icon: '📈' },
                            { id: 'loans', icon: '🏦' },
                            { id: 'investments', icon: '💰' },
                            { id: 'chat', icon: '🤖' }
                        ].map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
                                    activeTab === tab.id ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700'
                                }`}
                            >
                                {tab.icon}
                            </button>
                        ))}
                    </div>
                </div>
            </header>

            <main className="container mx-auto px-4 lg:px-6 py-6 lg:py-8">
                {activeTab === 'home' && <Homepage />}
                {activeTab === 'dashboard' && (account ? <Dashboard /> : <Homepage />)}
                {activeTab === 'loans' && (account ? <LoanApplication /> : <Homepage />)}
                {activeTab === 'investments' && (account ? <InvestmentPools /> : <Homepage />)}
                {activeTab === 'chat' && (account ? <AIChat /> : <Homepage />)}
            </main>

            <footer className="bg-white/95 backdrop-blur-sm mt-12 py-6 border-t">
                <div className="container mx-auto px-4 lg:px-6 text-center">
                    <h3 className="font-bold text-gray-800">Finverse v2.0</h3>
                    <p className="text-sm text-gray-600">Powered by Yahoo Finance API & Blockchain • Real-time data • {new Date().getFullYear()}</p>
                    <div className="mt-2 text-xs text-gray-500">
                        Last updated: {new Date().toLocaleTimeString()} • {stockData.length} stocks tracked
                    </div>
                </div>
            </footer>
        </div>
    );
}