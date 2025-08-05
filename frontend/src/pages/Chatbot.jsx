import React, { useState, useEffect, useRef } from 'react';
import { apiService } from '../services/api';

const Chatbot = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm your AI financial assistant. I can help you with loan applications, credit scores, investment advice, fraud reporting, and account inquiries. How can I assist you today?",
      sender: 'ai',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [userId] = useState(`user_${Date.now()}`); // Generate a unique user ID
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const quickQuestions = [
    "Check my loan status",
    "What's my credit score?", 
    "Give me investment advice for AAPL",
    "I want to report a fraud",
    "Show my account balance",
    "What services do you offer?"
  ];

  const sendMessage = async (messageText = inputMessage) => {
    if (!messageText.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: messageText,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);

    try {
      const response = await apiService.sendChatMessage(userId, messageText);
      
      const aiMessage = {
        id: Date.now() + 1,
        text: response.response,
        sender: 'ai',
        timestamp: new Date(),
        confidence: response.confidence
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "I apologize, but I'm having trouble processing your request right now. Please try again in a moment.",
        sender: 'ai',
        timestamp: new Date(),
        error: true
      };

      setMessages(prev => [...prev, errorMessage]);
      console.error('Chat error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage();
  };

  const handleQuickQuestion = (question) => {
    sendMessage(question);
  };

  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const clearChat = () => {
    setMessages([
      {
        id: 1,
        text: "Chat cleared! How can I help you today?",
        sender: 'ai',
        timestamp: new Date()
      }
    ]);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-4xl font-bold text-base-content mb-4">
            🤖 AI Financial Assistant
          </h1>
          <p className="text-lg text-base-content/70">
            Get instant help with loans, investments, fraud reporting, and more
          </p>
        </div>

        {/* Chat Container */}
        <div className="card bg-base-100 shadow-xl">
          {/* Chat Header */}
          <div className="card-body p-0">
            <div className="flex justify-between items-center p-4 border-b border-base-300">
              <div className="flex items-center gap-3">
                <div className="avatar">
                  <div className="w-10 rounded-full bg-primary text-primary-content flex items-center justify-center">
                    <span className="text-lg">🤖</span>
                  </div>
                </div>
                <div>
                  <h3 className="font-bold">AI Assistant</h3>
                  <p className="text-sm text-base-content/60">
                    <span className="inline-block w-2 h-2 bg-success rounded-full mr-1"></span>
                    Online
                  </p>
                </div>
              </div>
              <div className="flex gap-2">
                <button 
                  onClick={clearChat}
                  className="btn btn-ghost btn-sm"
                  title="Clear chat"
                >
                  🗑️
                </button>
              </div>
            </div>

            {/* Messages Area */}
            <div className="h-96 overflow-y-auto p-4 space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`chat ${message.sender === 'user' ? 'chat-end' : 'chat-start'}`}
                >
                  <div className="chat-image avatar">
                    <div className="w-8 rounded-full">
                      {message.sender === 'user' ? (
                        <div className="w-8 h-8 bg-secondary text-secondary-content rounded-full flex items-center justify-center">
                          👤
                        </div>
                      ) : (
                        <div className="w-8 h-8 bg-primary text-primary-content rounded-full flex items-center justify-center">
                          🤖
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="chat-header">
                    {message.sender === 'user' ? 'You' : 'AI Assistant'}
                    <time className="text-xs opacity-50 ml-1">
                      {formatTime(message.timestamp)}
                    </time>
                  </div>
                  <div className={`chat-bubble ${
                    message.sender === 'user' 
                      ? 'chat-bubble-primary' 
                      : message.error 
                        ? 'chat-bubble-error'
                        : 'chat-bubble-secondary'
                  }`}>
                    {message.text}
                    {message.confidence && (
                      <div className="text-xs opacity-70 mt-1">
                        Confidence: {(message.confidence * 100).toFixed(0)}%
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="chat chat-start">
                  <div className="chat-image avatar">
                    <div className="w-8 h-8 bg-primary text-primary-content rounded-full flex items-center justify-center">
                      🤖
                    </div>
                  </div>
                  <div className="chat-bubble chat-bubble-secondary">
                    <div className="flex items-center gap-2">
                      <span className="loading loading-dots loading-sm"></span>
                      Thinking...
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Quick Questions */}
            <div className="p-4 border-t border-base-300">
              <p className="text-sm text-base-content/60 mb-3">Quick questions:</p>
              <div className="flex flex-wrap gap-2">
                {quickQuestions.map((question, index) => (
                  <button
                    key={index}
                    onClick={() => handleQuickQuestion(question)}
                    disabled={loading}
                    className="btn btn-outline btn-xs"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-base-300">
              <form onSubmit={handleSubmit} className="flex gap-2">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder="Type your message here..."
                  className="input input-bordered flex-1"
                  disabled={loading}
                />
                <button
                  type="submit"
                  disabled={loading || !inputMessage.trim()}
                  className="btn btn-primary"
                >
                  {loading ? (
                    <span className="loading loading-spinner loading-sm"></span>
                  ) : (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                    </svg>
                  )}
                </button>
              </form>
            </div>
          </div>
        </div>

        {/* Features Info */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-8">
          <div className="card bg-base-100 shadow-lg">
            <div className="card-body text-center">
              <div className="text-3xl mb-2">💰</div>
              <h4 className="font-bold">Loan Services</h4>
              <p className="text-sm text-base-content/60">
                Check loan status, eligibility, and application processes
              </p>
            </div>
          </div>

          <div className="card bg-base-100 shadow-lg">
            <div className="card-body text-center">
              <div className="text-3xl mb-2">📊</div>
              <h4 className="font-bold">Credit Analysis</h4>
              <p className="text-sm text-base-content/60">
                Get credit score insights and improvement suggestions
              </p>
            </div>
          </div>

          <div className="card bg-base-100 shadow-lg">
            <div className="card-body text-center">
              <div className="text-3xl mb-2">📈</div>
              <h4 className="font-bold">Investment Advice</h4>
              <p className="text-sm text-base-content/60">
                Receive AI-powered investment recommendations
              </p>
            </div>
          </div>

          <div className="card bg-base-100 shadow-lg">
            <div className="card-body text-center">
              <div className="text-3xl mb-2">🛡️</div>
              <h4 className="font-bold">Fraud Protection</h4>
              <p className="text-sm text-base-content/60">
                Report suspicious activities and get security advice
              </p>
            </div>
          </div>

          <div className="card bg-base-100 shadow-lg">
            <div className="card-body text-center">
              <div className="text-3xl mb-2">💳</div>
              <h4 className="font-bold">Account Services</h4>
              <p className="text-sm text-base-content/60">
                Check balances, transactions, and account information
              </p>
            </div>
          </div>

          <div className="card bg-base-100 shadow-lg">
            <div className="card-body text-center">
              <div className="text-3xl mb-2">📋</div>
              <h4 className="font-bold">Compliance</h4>
              <p className="text-sm text-base-content/60">
                Get help with KYC, AML, and regulatory requirements
              </p>
            </div>
          </div>
        </div>

        {/* Privacy Notice */}
        <div className="alert alert-info mt-6">
          <svg className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          <div>
            <h4 className="font-bold">Privacy & Security</h4>
            <div className="text-xs">
              Your conversations are logged securely and hashed on our blockchain for audit purposes. 
              We do not store sensitive personal information in chat logs.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
