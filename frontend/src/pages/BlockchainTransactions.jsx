import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';

const BlockchainTransactions = () => {
  const [transactions, setTransactions] = useState([]);
  const [blockchainStats, setBlockchainStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [userFilter, setUserFilter] = useState('');
  const [typeFilter, setTypeFilter] = useState('');

  const fetchData = async () => {
    try {
      setError(null);
      const [transactionsResponse, statsResponse] = await Promise.all([
        apiService.getTransactions(50),
        apiService.getBlockchainStats()
      ]);
      
      setTransactions(transactionsResponse.transactions);
      setBlockchainStats(statsResponse.blockchain_stats);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch blockchain data');
      console.error('Blockchain data error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const getTransactionTypeIcon = (type) => {
    const icons = {
      'LOAN_DECISION': '💰',
      'FRAUD_ALERT': '🚨',
      'COMPLIANCE_ALERT': '⚖️',
      'CHAT_LOG': '💬'
    };
    return icons[type] || '📄';
  };

  const getTransactionTypeColor = (type) => {
    const colors = {
      'LOAN_DECISION': 'badge-primary',
      'FRAUD_ALERT': 'badge-error',
      'COMPLIANCE_ALERT': 'badge-warning',
      'CHAT_LOG': 'badge-info'
    };
    return colors[type] || 'badge-neutral';
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const truncateHash = (hash) => {
    if (!hash) return 'N/A';
    return `${hash.substring(0, 8)}...${hash.substring(hash.length - 8)}`;
  };

  const filteredTransactions = transactions.filter(tx => {
    const matchesUser = !userFilter || tx.user_id?.toLowerCase().includes(userFilter.toLowerCase());
    const matchesType = !typeFilter || tx.type === typeFilter;
    return matchesUser && matchesType;
  });

  const getUniqueTypes = () => {
    return [...new Set(transactions.map(tx => tx.type))];
  };

  const openTransactionModal = (transaction) => {
    setSelectedTransaction(transaction);
    document.getElementById('transaction_modal').showModal();
  };

  if (loading) {
    return <LoadingSpinner message="Loading blockchain data..." />;
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold text-base-content mb-2">
              🔗 Blockchain Transactions
            </h1>
            <p className="text-lg text-base-content/70">
              Immutable ledger of all platform activities
            </p>
          </div>
          
          <button
            onClick={fetchData}
            disabled={loading}
            className="btn btn-primary mt-4 sm:mt-0"
          >
            {loading ? (
              <span className="loading loading-spinner loading-sm"></span>
            ) : (
              '🔄'
            )}
            Refresh
          </button>
        </div>

        {/* Error State */}
        {error && (
          <div className="alert alert-error mb-6">
            <svg className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span>{error}</span>
          </div>
        )}

        {/* Blockchain Stats */}
        {blockchainStats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <div className="stat bg-base-100 rounded-lg shadow">
              <div className="stat-figure text-primary">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
                </svg>
              </div>
              <div className="stat-title">Total Blocks</div>
              <div className="stat-value text-primary">{blockchainStats.total_blocks}</div>
              <div className="stat-desc">In blockchain</div>
            </div>

            <div className="stat bg-base-100 rounded-lg shadow">
              <div className="stat-figure text-warning">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
              </div>
              <div className="stat-title">Pending</div>
              <div className="stat-value text-warning">{blockchainStats.pending_transactions}</div>
              <div className="stat-desc">Transactions</div>
            </div>

            <div className="stat bg-base-100 rounded-lg shadow">
              <div className="stat-figure text-success">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
              </div>
              <div className="stat-title">Chain Status</div>
              <div className={`stat-value ${blockchainStats.is_valid ? 'text-success' : 'text-error'}`}>
                {blockchainStats.is_valid ? 'Valid' : 'Invalid'}
              </div>
              <div className="stat-desc">Integrity check</div>
            </div>

            <div className="stat bg-base-100 rounded-lg shadow">
              <div className="stat-figure text-info">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                </svg>
              </div>
              <div className="stat-title">Difficulty</div>
              <div className="stat-value text-info">{blockchainStats.difficulty}</div>
              <div className="stat-desc">Mining difficulty</div>
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="card bg-base-100 shadow-lg mb-6">
          <div className="card-body">
            <h3 className="card-title mb-4">Filters</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="form-control">
                <label className="label">
                  <span className="label-text">Filter by User ID</span>
                </label>
                <input
                  type="text"
                  placeholder="Enter user ID..."
                  className="input input-bordered"
                  value={userFilter}
                  onChange={(e) => setUserFilter(e.target.value)}
                />
              </div>
              
              <div className="form-control">
                <label className="label">
                  <span className="label-text">Filter by Type</span>
                </label>
                <select
                  className="select select-bordered"
                  value={typeFilter}
                  onChange={(e) => setTypeFilter(e.target.value)}
                >
                  <option value="">All Types</option>
                  {getUniqueTypes().map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>
            </div>
            
            {(userFilter || typeFilter) && (
              <div className="mt-4">
                <span className="text-sm text-base-content/60">
                  Showing {filteredTransactions.length} of {transactions.length} transactions
                </span>
                <button
                  onClick={() => {
                    setUserFilter('');
                    setTypeFilter('');
                  }}
                  className="btn btn-ghost btn-sm ml-2"
                >
                  Clear Filters
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Transactions Table */}
        <div className="card bg-base-100 shadow-xl">
          <div className="card-body p-0">
            <div className="overflow-x-auto">
              <table className="table">
                <thead>
                  <tr className="bg-base-200">
                    <th>Type</th>
                    <th>User ID</th>
                    <th>Details</th>
                    <th>Timestamp</th>
                    <th>Transaction ID</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTransactions.map((transaction, index) => (
                    <tr key={`${transaction.transaction_id || index}`} className="hover:bg-base-50 transition-colors">
                      <td>
                        <div className="flex items-center gap-2">
                          <span className="text-xl">{getTransactionTypeIcon(transaction.type)}</span>
                          <span className={`badge ${getTransactionTypeColor(transaction.type)} badge-sm`}>
                            {transaction.type?.replace('_', ' ')}
                          </span>
                        </div>
                      </td>
                      
                      <td>
                        <span className="font-mono text-sm">
                          {transaction.user_id || 'N/A'}
                        </span>
                      </td>
                      
                      <td>
                        <div className="max-w-xs">
                          {transaction.type === 'LOAN_DECISION' && (
                            <div>
                              <div className={`badge ${transaction.approved ? 'badge-success' : 'badge-error'} badge-xs`}>
                                {transaction.approved ? 'Approved' : 'Rejected'}
                              </div>
                              <div className="text-xs text-base-content/60 mt-1">
                                Amount: ₹{transaction.amount?.toLocaleString()}
                              </div>
                            </div>
                          )}
                          
                          {transaction.type === 'FRAUD_ALERT' && (
                            <div>
                              <div className="badge badge-error badge-xs">
                                Fraud Risk: {(transaction.fraud_probability * 100)?.toFixed(1)}%
                              </div>
                              <div className="text-xs text-base-content/60 mt-1">
                                Amount: ₹{transaction.amount?.toLocaleString()}
                              </div>
                            </div>
                          )}
                          
                          {transaction.type === 'COMPLIANCE_ALERT' && (
                            <div>
                              <div className={`badge ${transaction.compliant ? 'badge-success' : 'badge-warning'} badge-xs`}>
                                {transaction.compliant ? 'Compliant' : 'Violations Found'}
                              </div>
                              <div className="text-xs text-base-content/60 mt-1">
                                Risk: {transaction.risk_level}
                              </div>
                            </div>
                          )}
                          
                          {transaction.type === 'CHAT_LOG' && (
                            <div>
                              <div className="badge badge-info badge-xs">
                                Intent: {transaction.intent}
                              </div>
                              <div className="text-xs text-base-content/60 mt-1">
                                Hash: {transaction.chat_hash?.substring(0, 16)}...
                              </div>
                            </div>
                          )}
                        </div>
                      </td>
                      
                      <td>
                        <span className="text-xs">
                          {formatTimestamp(transaction.timestamp)}
                        </span>
                      </td>
                      
                      <td>
                        <span className="font-mono text-xs">
                          {truncateHash(transaction.transaction_id)}
                        </span>
                      </td>
                      
                      <td>
                        <button
                          onClick={() => openTransactionModal(transaction)}
                          className="btn btn-ghost btn-xs"
                        >
                          View Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              
              {filteredTransactions.length === 0 && (
                <div className="text-center py-12">
                  <div className="text-6xl mb-4">🔗</div>
                  <h3 className="text-2xl font-bold mb-2">No transactions found</h3>
                  <p className="text-base-content/60">
                    {userFilter || typeFilter 
                      ? 'No transactions match your current filters'
                      : 'No blockchain transactions available yet'
                    }
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Latest Block Hash */}
        {blockchainStats && (
          <div className="mt-6">
            <div className="alert alert-info">
              <svg className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              <div>
                <h4 className="font-bold">Latest Block Hash</h4>
                <div className="text-xs font-mono break-all">
                  {blockchainStats.latest_block_hash}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Transaction Detail Modal */}
        <dialog id="transaction_modal" className="modal">
          <div className="modal-box max-w-2xl">
            <form method="dialog">
              <button className="btn btn-sm btn-circle btn-ghost absolute right-2 top-2">✕</button>
            </form>
            
            {selectedTransaction && (
              <div>
                <h3 className="font-bold text-lg mb-4">
                  {getTransactionTypeIcon(selectedTransaction.type)} Transaction Details
                </h3>
                
                <div className="space-y-4">
                  <div>
                    <strong>Type:</strong>
                    <span className={`badge ${getTransactionTypeColor(selectedTransaction.type)} ml-2`}>
                      {selectedTransaction.type?.replace('_', ' ')}
                    </span>
                  </div>
                  
                  <div>
                    <strong>Transaction ID:</strong>
                    <div className="font-mono text-sm mt-1 p-2 bg-base-200 rounded break-all">
                      {selectedTransaction.transaction_id}
                    </div>
                  </div>
                  
                  <div>
                    <strong>User ID:</strong> {selectedTransaction.user_id || 'N/A'}
                  </div>
                  
                  <div>
                    <strong>Timestamp:</strong> {formatTimestamp(selectedTransaction.timestamp)}
                  </div>
                  
                  <div>
                    <strong>Raw Data:</strong>
                    <pre className="text-xs mt-1 p-3 bg-base-200 rounded overflow-auto max-h-60">
                      {JSON.stringify(selectedTransaction, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            )}
          </div>
          <form method="dialog" className="modal-backdrop">
            <button>close</button>
          </form>
        </dialog>
      </div>
    </div>
  );
};

export default BlockchainTransactions;
