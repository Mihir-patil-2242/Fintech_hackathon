import React, { useState } from 'react';
import { apiService } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';

const LoanApplication = () => {
  const [formData, setFormData] = useState({
    user_id: '',
    income: '',
    loan_amount: '',
    credit_history_score: 5,
    employment_years: '',
    existing_debts: '',
    purpose: ''
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Convert string values to numbers where needed
      const requestData = {
        ...formData,
        income: parseFloat(formData.income),
        loan_amount: parseFloat(formData.loan_amount),
        credit_history_score: parseInt(formData.credit_history_score),
        employment_years: parseInt(formData.employment_years),
        existing_debts: parseFloat(formData.existing_debts)
      };

      const response = await apiService.submitLoanRequest(requestData);
      setResult(response);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process loan application');
    } finally {
      setLoading(false);
    }
  };

  const clearForm = () => {
    setFormData({
      user_id: '',
      income: '',
      loan_amount: '',
      credit_history_score: 5,
      employment_years: '',
      existing_debts: '',
      purpose: ''
    });
    setResult(null);
    setError(null);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-base-content mb-4">
            💰 AI-Powered Loan Application
          </h1>
          <p className="text-lg text-base-content/70">
            Get instant loan decisions powered by AI and secured on blockchain
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Application Form */}
          <div className="card bg-base-100 shadow-xl">
            <div className="card-body">
              <h2 className="card-title text-2xl mb-6">Loan Application Form</h2>
              
              <form onSubmit={handleSubmit} className="space-y-4">
                {/* User ID */}
                <div className="form-control">
                  <label className="label">
                    <span className="label-text font-semibold">User ID</span>
                  </label>
                  <input
                    type="text"
                    name="user_id"
                    value={formData.user_id}
                    onChange={handleInputChange}
                    placeholder="Enter your unique user ID"
                    className="input input-bordered w-full"
                    required
                  />
                </div>

                {/* Income */}
                <div className="form-control">
                  <label className="label">
                    <span className="label-text font-semibold">Annual Income (₹)</span>
                  </label>
                  <input
                    type="number"
                    name="income"
                    value={formData.income}
                    onChange={handleInputChange}
                    placeholder="e.g., 500000"
                    className="input input-bordered w-full"
                    min="0"
                    step="1000"
                    required
                  />
                </div>

                {/* Loan Amount */}
                <div className="form-control">
                  <label className="label">
                    <span className="label-text font-semibold">Loan Amount (₹)</span>
                  </label>
                  <input
                    type="number"
                    name="loan_amount"
                    value={formData.loan_amount}
                    onChange={handleInputChange}
                    placeholder="e.g., 200000"
                    className="input input-bordered w-full"
                    min="0"
                    step="1000"
                    required
                  />
                </div>

                {/* Credit History Score */}
                <div className="form-control">
                  <label className="label">
                    <span className="label-text font-semibold">
                      Credit History Score (1-10): {formData.credit_history_score}
                    </span>
                  </label>
                  <input
                    type="range"
                    name="credit_history_score"
                    value={formData.credit_history_score}
                    onChange={handleInputChange}
                    min="1"
                    max="10"
                    className="range range-primary"
                  />
                  <div className="w-full flex justify-between text-xs px-2">
                    <span>Poor</span>
                    <span>Average</span>
                    <span>Excellent</span>
                  </div>
                </div>

                {/* Employment Years */}
                <div className="form-control">
                  <label className="label">
                    <span className="label-text font-semibold">Employment Years</span>
                  </label>
                  <input
                    type="number"
                    name="employment_years"
                    value={formData.employment_years}
                    onChange={handleInputChange}
                    placeholder="e.g., 5"
                    className="input input-bordered w-full"
                    min="0"
                    max="50"
                    required
                  />
                </div>

                {/* Existing Debts */}
                <div className="form-control">
                  <label className="label">
                    <span className="label-text font-semibold">Existing Debts (₹)</span>
                  </label>
                  <input
                    type="number"
                    name="existing_debts"
                    value={formData.existing_debts}
                    onChange={handleInputChange}
                    placeholder="e.g., 50000"
                    className="input input-bordered w-full"
                    min="0"
                    step="1000"
                    required
                  />
                </div>

                {/* Purpose */}
                <div className="form-control">
                  <label className="label">
                    <span className="label-text font-semibold">Loan Purpose</span>
                  </label>
                  <select
                    name="purpose"
                    value={formData.purpose}
                    onChange={handleInputChange}
                    className="select select-bordered w-full"
                    required
                  >
                    <option value="">Select loan purpose</option>
                    <option value="home">Home Purchase</option>
                    <option value="car">Car Purchase</option>
                    <option value="business">Business</option>
                    <option value="education">Education</option>
                    <option value="personal">Personal</option>
                    <option value="medical">Medical</option>
                  </select>
                </div>

                {/* Buttons */}
                <div className="flex gap-4 pt-4">
                  <button
                    type="submit"
                    disabled={loading}
                    className="btn btn-primary flex-1"
                  >
                    {loading ? (
                      <>
                        <span className="loading loading-spinner loading-sm"></span>
                        Processing...
                      </>
                    ) : (
                      '🚀 Submit Application'
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={clearForm}
                    className="btn btn-outline"
                  >
                    Clear
                  </button>
                </div>
              </form>
            </div>
          </div>

          {/* Results Panel */}
          <div className="card bg-base-100 shadow-xl">
            <div className="card-body">
              <h2 className="card-title text-2xl mb-6">Application Status</h2>
              
              {loading && (
                <LoadingSpinner message="Processing your loan application..." />
              )}

              {error && (
                <div className="alert alert-error">
                  <svg className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                  <span>{error}</span>
                </div>
              )}

              {result && (
                <div className="space-y-4">
                  {/* Main Result */}
                  <div className={`alert ${result.approved ? 'alert-success' : 'alert-error'}`}>
                    <svg className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" 
                            d={result.approved ? "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" : "M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"}></path>
                    </svg>
                    <div>
                      <h3 className="font-bold">
                        {result.approved ? '🎉 Loan Approved!' : '❌ Loan Rejected'}
                      </h3>
                      <div className="text-xs">{result.reason}</div>
                    </div>
                  </div>

                  {/* Loan Details */}
                  <div className="stats stats-vertical w-full">
                    <div className="stat">
                      <div className="stat-title">Loan ID</div>
                      <div className="stat-value text-lg">{result.loan_id}</div>
                    </div>
                    
                    <div className="stat">
                      <div className="stat-title">Credit Score</div>
                      <div className={`stat-value ${result.credit_score >= 700 ? 'text-success' : result.credit_score >= 600 ? 'text-warning' : 'text-error'}`}>
                        {result.credit_score}
                      </div>
                      <div className="stat-desc">
                        {result.credit_score >= 700 ? 'Excellent' : 
                         result.credit_score >= 600 ? 'Good' : 'Needs Improvement'}
                      </div>
                    </div>

                    {result.blockchain_tx_id && (
                      <div className="stat">
                        <div className="stat-title">Blockchain Record</div>
                        <div className="stat-value text-sm break-all">
                          {result.blockchain_tx_id.substring(0, 16)}...
                        </div>
                        <div className="stat-desc">
                          <span className="badge badge-info">🔗 Verified</span>
                        </div>
                      </div>
                    )}
                  </div>

                  {result.approved && (
                    <div className="card bg-success/10 border border-success/20">
                      <div className="card-body">
                        <h4 className="card-title text-success">Next Steps</h4>
                        <ul className="list-disc list-inside space-y-1 text-sm">
                          <li>Submit required documents</li>
                          <li>Complete KYC verification</li>
                          <li>Schedule property valuation (if applicable)</li>
                          <li>Loan disbursement within 7 days</li>
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {!loading && !result && !error && (
                <div className="text-center text-base-content/50 py-8">
                  <div className="text-6xl mb-4">📋</div>
                  <p>Fill out the application form to get started</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoanApplication;
