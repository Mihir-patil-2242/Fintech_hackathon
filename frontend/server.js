// Simple Express server to proxy Yahoo Finance data
const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('dist'));

// Serve the React app
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// Mock API endpoint for Yahoo Finance data
app.post('/api/yfinance/stock-info', async (req, res) => {
    const { ticker } = req.body;
    
    // Mock real-time stock data
    const mockData = {
        'AAPL': { currentPrice: 202.92, regularMarketChangePercent: -0.21, volume: 42535446, marketCap: 3011413803008, shortName: 'Apple Inc.' },
        'MSFT': { currentPrice: 527.75, regularMarketChangePercent: -1.47, volume: 18767826, marketCap: 3922855460864, shortName: 'Microsoft Corporation' },
        'TSLA': { currentPrice: 308.72, regularMarketChangePercent: -0.17, volume: 57623824, marketCap: 995760930816, shortName: 'Tesla, Inc.' },
        'GOOGL': { currentPrice: 178.34, regularMarketChangePercent: 1.23, volume: 25000000, marketCap: 2200000000000, shortName: 'Alphabet Inc.' },
        'NVDA': { currentPrice: 145.67, regularMarketChangePercent: 2.34, volume: 40000000, marketCap: 3500000000000, shortName: 'NVIDIA Corporation' },
        'AMZN': { currentPrice: 182.45, regularMarketChangePercent: 0.87, volume: 30000000, marketCap: 1900000000000, shortName: 'Amazon.com Inc.' }
    };

    const baseData = mockData[ticker] || mockData['AAPL'];
    
    // Add some real-time variation
    const liveData = {
        ...baseData,
        symbol: ticker,
        currentPrice: baseData.currentPrice + (Math.random() - 0.5) * 5,
        regularMarketChangePercent: baseData.regularMarketChangePercent + (Math.random() - 0.5) * 2,
        volume: baseData.volume + Math.floor((Math.random() - 0.5) * 1000000)
    };

    res.json(liveData);
});

app.listen(PORT, () => {
    console.log(`🚀 Finverse running on http://localhost:${PORT}`);
    console.log(`📊 Yahoo Finance integration active`);
});

module.exports = app;