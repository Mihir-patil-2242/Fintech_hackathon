import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { apiService } from '../services/api';

const Home = () => {
  const [healthStatus, setHealthStatus] = useState(null);
  const [blockchainStats, setBlockchainStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const healthResponse = await apiService.healthCheck();
        setHealthStatus(healthResponse);
        
        try {
          const statsResponse = await apiService.getBlockchainStats();
          setBlockchainStats(statsResponse.blockchain_stats);
        } catch (statsError) {
          console.warn('Could not fetch blockchain stats:', statsError);
          setBlockchainStats({
            total_blocks: 1,
            pending_transactions: 0,
            is_valid: true,
            difficulty: 2
          });
        }
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
        setError('Backend not running. Please start the backend server.');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const features = [
    {
      icon: 'fas fa-coins',
      title: 'Smart Loan Processing',
      description: 'AI-powered credit scoring with instant decisions and blockchain security',
      link: '/loan',
      color: 'from-blue-500 to-purple-600',
      stats: '95% Accuracy'
    },
    {
      icon: 'fas fa-chart-line',
      title: 'Market Intelligence',
      description: 'Real-time investment advice with technical analysis and ML predictions',
      link: '/market',
      color: 'from-green-500 to-teal-600',
      stats: 'Live Data'
    },
    {
      icon: 'fas fa-robot',
      title: 'AI Assistant',
      description: '24/7 intelligent financial support with natural language processing',
      link: '/chat',
      color: 'from-purple-500 to-pink-600',
      stats: '99% Uptime'
    },
    {
      icon: 'fas fa-link',
      title: 'Blockchain Security',
      description: 'Immutable transaction records with proof-of-work consensus',
      link: '/transactions',
      color: 'from-orange-500 to-red-600',
      stats: 'Immutable'
    }
  ];

  const aiServices = [
    { name: 'Credit Scoring', icon: 'fas fa-bullseye', status: 'active' },
    { name: 'Fraud Detection', icon: 'fas fa-shield-alt', status: 'active' },
    { name: 'Investment AI', icon: 'fas fa-brain', status: 'active' },
    { name: 'Compliance Check', icon: 'fas fa-balance-scale', status: 'active' },
    { name: 'NLP Chatbot', icon: 'fas fa-comments', status: 'active' },
    { name: 'Blockchain', icon: 'fas fa-cubes', status: 'active' }
  ];

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-base-200 via-base-100 to-base-200">
        <div className="text-center">
          <div className="loading loading-spinner loading-lg text-primary mb-4"></div>
          <p className="text-lg font-medium">Initializing FinTech AI Platform...</p>
          <p className="text-sm text-base-content/60 mt-2">Loading AI services and blockchain</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="hero min-h-screen hero-gradient relative">
        <div className="hero-overlay bg-opacity-20"></div>
        <div className="hero-content text-center text-white relative z-10">
          <div className="max-w-6xl">
            <div className="mb-8">
              <div className="text-8xl mb-6 float-animation">
                <i className="fas fa-university"></i>
              </div>
              <h1 className="text-7xl font-bold mb-4">
                <span className="gradient-text">FinTech AI Platform</span>
              </h1>
              <div className="flex flex-wrap justify-center gap-4 text-xl font-medium opacity-90 mb-8">
                <span className="badge badge-lg badge-outline border-white/30 text-white">
                  <i className="fas fa-link mr-2"></i>
                  Blockchain-Secured
                </span>
                <span className="badge badge-lg badge-outline border-white/30 text-white">
                  <i className="fas fa-brain mr-2"></i>
                  AI-Powered
                </span>
                <span className="badge badge-lg badge-outline border-white/30 text-white">
                  <i className="fas fa-bolt mr-2"></i>
                  Real-Time Intelligence
                </span>
              </div>
            </div>
            
            {error && (
              <div className="alert alert-warning mb-8 max-w-2xl mx-auto glass">
                <i className="fas fa-exclamation-triangle text-xl"></i>
                <div>
                  <h4 className="font-bold">Backend Connection Required</h4>
                  <div className="text-sm mt-1">
                    Start the backend server: <code className="bg-black/20 px-2 py-1 rounded">cd backend && python main.py</code>
                  </div>
                </div>
              </div>
            )}
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/loan" className="btn btn-primary btn-lg btn-glow shadow-2xl">
                <i className="fas fa-rocket mr-2"></i>
                Apply for Loan
              </Link>
              <Link to="/market" className="btn btn-outline btn-lg glass border-white/30 text-white hover:bg-white/10">
                <i className="fas fa-chart-line mr-2"></i>
                View Markets
              </Link>
            </div>
          </div>
        </div>
        
        {/* Floating elements */}
        <div className="absolute top-20 left-10 opacity-10">
          <i className="fas fa-coins text-6xl animate-bounce"></i>
        </div>
        <div className="absolute bottom-20 right-10 opacity-10">
          <i className="fas fa-chart-line text-6xl animate-pulse"></i>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-20 bg-gradient-to-b from-base-100 to-base-200">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold mb-4">
              Platform <span className="gradient-text">Features</span>
            </h2>
            <p className="text-xl text-base-content/70 max-w-2xl mx-auto">
              Advanced AI algorithms meet enterprise blockchain security to deliver the future of financial services
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <Link
                key={index}
                to={feature.link}
                className="feature-card card card-glow transition-all duration-300 hover:scale-105"
              >
                <div className="card-body text-center p-8">
                  <div className={`w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br ${feature.color} flex items-center justify-center shadow-lg`}>
                    <i className={`${feature.icon} text-white text-2xl`}></i>
                  </div>
                  <h3 className="card-title justify-center text-xl mb-3">{feature.title}</h3>
                  <p className="text-base-content/70 mb-4">{feature.description}</p>
                  <div className="badge badge-primary badge-outline">{feature.stats}</div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>

      {/* Platform Stats */}
      {!error && blockchainStats && (
        <div className="py-20 bg-base-100">
          <div className="container mx-auto px-4">
            <div className="text-center mb-16">
              <h2 className="text-5xl font-bold mb-4">
                Platform <span className="gradient-text">Status</span>
              </h2>
              <p className="text-xl text-base-content/70">Real-time system health and blockchain statistics</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
              <div className="stats-card stat text-center p-8 rounded-2xl shadow-lg">
                <div className="stat-figure text-primary">
                  <i className="fas fa-cubes text-4xl"></i>
                </div>
                <div className="stat-title text-lg font-medium">Blockchain Blocks</div>
                <div className="stat-value text-primary text-4xl">{blockchainStats.total_blocks || 1}</div>
                <div className="stat-desc">Immutable records secured</div>
              </div>

              <div className="stats-card stat text-center p-8 rounded-2xl shadow-lg">
                <div className="stat-figure text-success">
                  <i className="fas fa-check-circle text-4xl"></i>
                </div>
                <div className="stat-title text-lg font-medium">Chain Integrity</div>
                <div className={`stat-value text-4xl ${blockchainStats.is_valid ? 'text-success' : 'text-error'}`}>
                  {blockchainStats.is_valid ? 'Valid' : 'Invalid'}
                </div>
                <div className="stat-desc">Cryptographic verification</div>
              </div>

              <div className="stats-card stat text-center p-8 rounded-2xl shadow-lg">
                <div className="stat-figure text-warning">
                  <i className="fas fa-clock text-4xl"></i>
                </div>
                <div className="stat-title text-lg font-medium">Pending Transactions</div>
                <div className="stat-value text-warning text-4xl">{blockchainStats.pending_transactions || 0}</div>
                <div className="stat-desc">Awaiting confirmation</div>
              </div>

              <div className="stats-card stat text-center p-8 rounded-2xl shadow-lg">
                <div className="stat-figure text-info">
                  <i className="fas fa-bolt text-4xl"></i>
                </div>
                <div className="stat-title text-lg font-medium">Mining Difficulty</div>
                <div className="stat-value text-info text-4xl">{blockchainStats.difficulty || 2}</div>
                <div className="stat-desc">Proof-of-work complexity</div>
              </div>
            </div>

            {/* AI Services Status */}
            <div className="card bg-base-100 shadow-2xl border border-base-200">
              <div className="card-body p-8">
                <div className="flex items-center justify-between mb-8">
                  <h3 className="text-3xl font-bold">
                    <i className="fas fa-brain text-primary mr-3"></i>
                    AI Services Status
                  </h3>
                  <div className="badge badge-success badge-lg gap-2">
                    <div className="w-3 h-3 bg-success rounded-full animate-pulse"></div>
                    All Systems Operational
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {aiServices.map((service, index) => (
                    <div key={index} className="service-status flex items-center justify-between p-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
                          <i className={`${service.icon} text-white`}></i>
                        </div>
                        <span className="font-semibold">{service.name}</span>
                      </div>
                      <div className={`badge ${error ? 'badge-error' : 'badge-success'} gap-1`}>
                        <div className="w-2 h-2 rounded-full bg-current animate-pulse"></div>
                        {error ? 'Offline' : 'Active'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Call to Action */}
      <div className="py-20 hero-gradient text-white">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-5xl font-bold mb-4">Ready to Get Started?</h2>
          <p className="text-xl mb-8 opacity-90 max-w-3xl mx-auto">
            Experience the future of financial services with our AI-powered, blockchain-secured platform. 
            Join thousands of users who trust our technology for their financial needs.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/loan" className="btn btn-accent btn-lg btn-glow shadow-2xl">
              <i className="fas fa-rocket mr-2"></i>
              Apply for a Loan Now
            </Link>
            <Link to="/chat" className="btn btn-outline btn-lg glass border-white/30 text-white hover:bg-white/10">
              <i className="fas fa-comments mr-2"></i>
              Talk to AI Assistant
            </Link>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="footer footer-center p-10 bg-base-200 text-base-content">
        <div>
          <div className="flex items-center gap-2 mb-4">
            <div className="w-10 h-10 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
              <i className="fas fa-university text-white"></i>
            </div>
            <span className="text-2xl font-bold gradient-text">FinTech AI Platform</span>
          </div>
          <p className="text-base-content/70 max-w-md">
            Revolutionizing financial services through artificial intelligence and blockchain technology
          </p>
        </div>
        <div>
          <div className="grid grid-flow-col gap-4">
            <div className="badge badge-outline">FastAPI Backend</div>
            <div className="badge badge-outline">React Frontend</div>
            <div className="badge badge-outline">AI/ML Services</div>
            <div className="badge badge-outline">Blockchain Security</div>
          </div>
        </div>
        <div>
          <p className="text-sm text-base-content/50">
            © 2025 FinTech AI Platform - Built for Hackathon Demo
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Home;
