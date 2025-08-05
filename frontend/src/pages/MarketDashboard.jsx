import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';

const MarketDashboard = () => {
  const [marketData, setMarketData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchMarketData = async () => {
    try {
      setError(null);
      const response = await apiService.getPopularStocks();
      setMarketData(response.data);
      setLastUpdated(new Date(response.timestamp));
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch market data');
      console.error('Market data error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMarketData();
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchMarketData, 30000); // Refresh every 30 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getRecommendationColor = (recommendation) => {
    const rec = recommendation.toLowerCase();
    if (rec.includes('buy')) return 'text-success';
    if (rec.includes('sell')) return 'text-error';
    return 'text-warning';
  };

  const getRecommendationBadge = (recommendation) => {
    const rec = recommendation.toLowerCase();
    if (rec.includes('strong buy')) return 'badge-success';
    if (rec.includes('buy')) return 'badge-info';
    if (rec.includes('strong sell')) return 'badge-error';
    if (rec.includes('sell')) return 'badge-warning';
    return 'badge-neutral';
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 2
    }).format(price);
  };

  const formatVolume = (volume) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    }
    return volume.toString();
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold text-base-content mb-2">
              📈 Live Market Dashboard
            </h1>
            <p className="text-lg text-base-content/70">
              AI-powered investment recommendations with real-time data
            </p>
          </div>
          
          <div className="flex items-center gap-4 mt-4 sm:mt-0">
            <div className="form-control">
              <label className="label cursor-pointer">
                <span className="label-text mr-2">Auto Refresh</span>
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="toggle toggle-primary"
                />
              </label>
            </div>
            
            <button
              onClick={fetchMarketData}
              disabled={loading}
              className="btn btn-primary btn-sm"
            >
              {loading ? (
                <span className="loading loading-spinner loading-sm"></span>
              ) : (
                '🔄'
              )}
              Refresh
            </button>
          </div>
        </div>

        {/* Last Updated */}
        {lastUpdated && (
          <div className="alert alert-info mb-6">
            <svg className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span>Last updated: {lastUpdated.toLocaleString()}</span>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="alert alert-error mb-6">
            <svg className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span>{error}</span>
          </div>
        )}

        {/* Loading State */}
        {loading && marketData.length === 0 ? (
          <LoadingSpinner message="Fetching live market data..." />
        ) : (
          <>
            {/* Market Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              <div className="stat bg-base-100 rounded-lg shadow">
                <div className="stat-figure text-primary">
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
                  </svg>
                </div>
                <div className="stat-title">Total Stocks</div>
                <div className="stat-value text-primary">{marketData.length}</div>
                <div className="stat-desc">Being monitored</div>
              </div>

              <div className="stat bg-base-100 rounded-lg shadow">
                <div className="stat-figure text-success">
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 11l5-5m0 0l5 5m-5-5v12"></path>
                  </svg>
                </div>
                <div className="stat-title">Buy Signals</div>
                <div className="stat-value text-success">
                  {marketData.filter(stock => stock.recommendation?.toLowerCase().includes('buy')).length}
                </div>
                <div className="stat-desc">AI recommendations</div>
              </div>

              <div className="stat bg-base-100 rounded-lg shadow">
                <div className="stat-figure text-error">
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 13l-5 5m0 0l-5-5m5 5V6"></path>
                  </svg>
                </div>
                <div className="stat-title">Sell Signals</div>
                <div className="stat-value text-error">
                  {marketData.filter(stock => stock.recommendation?.toLowerCase().includes('sell')).length}
                </div>
                <div className="stat-desc">AI recommendations</div>
              </div>

              <div className="stat bg-base-100 rounded-lg shadow">
                <div className="stat-figure text-warning">
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                </div>
                <div className="stat-title">Hold Signals</div>
                <div className="stat-value text-warning">
                  {marketData.filter(stock => stock.recommendation?.toLowerCase().includes('hold')).length}
                </div>
                <div className="stat-desc">Wait and watch</div>
              </div>
            </div>

            {/* Stock Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {marketData.map((stock, index) => (
                <div key={`${stock.symbol}-${index}`} className="card bg-base-100 shadow-xl hover:shadow-2xl transition-shadow duration-300">
                  <div className="card-body">
                    {/* Stock Header */}
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h3 className="card-title text-xl">{stock.symbol}</h3>
                        <p className="text-base-content/60 text-sm">
                          {stock.symbol.includes('.NS') ? 'NSE' : 'NASDAQ'}
                        </p>
                      </div>
                      <div className={`badge ${getRecommendationBadge(stock.recommendation)} badge-lg`}>
                        {stock.recommendation}
                      </div>
                    </div>

                    {/* Price Info */}
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="stat p-0">
                        <div className="stat-title text-xs">Current Price</div>
                        <div className="stat-value text-lg">
                          {formatPrice(stock.current_price)}
                        </div>
                      </div>
                      <div className="stat p-0">
                        <div className="stat-title text-xs">Change</div>
                        <div className={`stat-value text-lg ${stock.change_percent >= 0 ? 'text-success' : 'text-error'}`}>
                          {stock.change_percent >= 0 ? '+' : ''}{stock.change_percent}%
                        </div>
                      </div>
                    </div>

                    {/* Technical Indicators */}
                    <div className="space-y-2 mb-4">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-base-content/70">7-Day MA</span>
                        <span className="font-mono text-sm">{formatPrice(stock.ma_7)}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-base-content/70">RSI</span>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-sm">{stock.rsi?.toFixed(1)}</span>
                          <div className={`badge badge-xs ${
                            stock.rsi < 30 ? 'badge-success' : 
                            stock.rsi > 70 ? 'badge-error' : 
                            'badge-warning'
                          }`}>
                            {stock.rsi < 30 ? 'Oversold' : stock.rsi > 70 ? 'Overbought' : 'Neutral'}
                          </div>
                        </div>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-base-content/70">Volume</span>
                        <span className="font-mono text-sm">{formatVolume(stock.volume)}</span>
                      </div>
                    </div>

                    {/* RSI Progress Bar */}
                    <div className="mb-4">
                      <div className="flex justify-between text-xs mb-1">
                        <span>RSI: {stock.rsi?.toFixed(1)}</span>
                        <span className={
                          stock.rsi < 30 ? 'text-success' : 
                          stock.rsi > 70 ? 'text-error' : 
                          'text-warning'
                        }>
                          {stock.rsi < 30 ? 'Buy Zone' : stock.rsi > 70 ? 'Sell Zone' : 'Neutral'}
                        </span>
                      </div>
                      <progress 
                        className={`progress w-full ${
                          stock.rsi < 30 ? 'progress-success' : 
                          stock.rsi > 70 ? 'progress-error' : 
                          'progress-warning'
                        }`}
                        value={stock.rsi} 
                        max="100"
                      ></progress>
                    </div>

                    {/* Action Buttons */}
                    <div className="card-actions justify-end">
                      <button className="btn btn-outline btn-sm">
                        📊 Details
                      </button>
                      <button className={`btn btn-sm ${
                        stock.recommendation?.toLowerCase().includes('buy') ? 'btn-success' :
                        stock.recommendation?.toLowerCase().includes('sell') ? 'btn-error' :
                        'btn-warning'
                      }`}>
                        {stock.recommendation?.toLowerCase().includes('buy') ? '📈 Buy' :
                         stock.recommendation?.toLowerCase().includes('sell') ? '📉 Sell' :
                         '⏸️ Hold'}
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Empty State */}
            {marketData.length === 0 && !loading && (
              <div className="text-center py-12">
                <div className="text-6xl mb-4">📈</div>
                <h3 className="text-2xl font-bold mb-2">No market data available</h3>
                <p className="text-base-content/60 mb-6">
                  Unable to fetch market data at the moment. Please try refreshing.
                </p>
                <button onClick={fetchMarketData} className="btn btn-primary">
                  🔄 Retry
                </button>
              </div>
            )}

            {/* Disclaimer */}
            <div className="alert alert-warning mt-8">
              <svg className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
              </svg>
              <div>
                <h4 className="font-bold">Investment Disclaimer</h4>
                <div className="text-xs">
                  This is a hackathon demo. AI recommendations are for demonstration purposes only and should not be considered as actual investment advice. 
                  Always consult with qualified financial advisors before making investment decisions.
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default MarketDashboard;
