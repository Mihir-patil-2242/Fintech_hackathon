import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export const apiService = {
  // Loan requests
  submitLoanRequest: async (loanData) => {
    const response = await api.post('/loan-request', loanData);
    return response.data;
  },

  // Fraud detection
  checkFraud: async (transactionData) => {
    const response = await api.post('/check-fraud', transactionData);
    return response.data;
  },

  // Compliance check
  checkCompliance: async (complianceData) => {
    const response = await api.post('/compliance-check', complianceData);
    return response.data;
  },

  // Market data
  getMarketData: async (symbols = 'AAPL,GOOGL,MSFT,TSLA,RELIANCE.NS') => {
    const response = await api.get(`/market-data?symbols=${symbols}`);
    return response.data;
  },

  getPopularStocks: async () => {
    const response = await api.get('/market-data/popular');
    return response.data;
  },

  // Chatbot
  sendChatMessage: async (userId, message) => {
    const response = await api.post('/chatbot', {
      user_id: userId,
      message: message
    });
    return response.data;
  },

  // Blockchain transactions
  getTransactions: async (limit = 20) => {
    const response = await api.get(`/transactions?limit=${limit}`);
    return response.data;
  },

  getUserTransactions: async (userId) => {
    const response = await api.get(`/transactions/user/${userId}`);
    return response.data;
  },

  getBlockchainStats: async () => {
    const response = await api.get('/blockchain/stats');
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  }
};

export default apiService;
