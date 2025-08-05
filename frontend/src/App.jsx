import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import LoanApplication from './pages/LoanApplication';
import MarketDashboard from './pages/MarketDashboard';
import Chatbot from './pages/Chatbot';
import BlockchainTransactions from './pages/BlockchainTransactions';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-base-100">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/loan" element={<LoanApplication />} />
            <Route path="/market" element={<MarketDashboard />} />
            <Route path="/chat" element={<Chatbot />} />
            <Route path="/transactions" element={<BlockchainTransactions />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
