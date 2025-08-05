import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar = () => {
  const location = useLocation();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const isActive = (path) => {
    return location.pathname === path;
  };

  const navItems = [
    { path: '/loan', icon: 'fas fa-coins', label: 'Loan Application' },
    { path: '/market', icon: 'fas fa-chart-line', label: 'Market Dashboard' },
    { path: '/chat', icon: 'fas fa-robot', label: 'AI Assistant' },
    { path: '/transactions', icon: 'fas fa-link', label: 'Blockchain' },
  ];

  return (
    <>
      <div className="navbar bg-base-100/95 backdrop-blur-md shadow-lg sticky top-0 z-50 border-b border-base-200">
        <div className="navbar-start">
          <div className="dropdown lg:hidden">
            <div 
              tabIndex={0} 
              role="button" 
              className="btn btn-ghost"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h8m-8 6h16"></path>
              </svg>
            </div>
            {isMenuOpen && (
              <ul 
                tabIndex={0} 
                className="menu menu-sm dropdown-content mt-3 z-[1] p-2 shadow-xl bg-base-100 rounded-box w-52 border border-base-200"
              >
                {navItems.map((item) => (
                  <li key={item.path}>
                    <Link 
                      to={item.path}
                      className={`${isActive(item.path) ? 'active bg-primary text-primary-content' : ''}`}
                      onClick={() => setIsMenuOpen(false)}
                    >
                      <i className={item.icon}></i>
                      {item.label}
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </div>
          
          <Link to="/" className="btn btn-ghost text-xl font-bold">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
                <i className="fas fa-university text-white text-sm"></i>
              </div>
              <span className="gradient-text">FinTech AI</span>
            </div>
          </Link>
        </div>
        
        <div className="navbar-center hidden lg:flex">
          <ul className="menu menu-horizontal px-1 gap-1">
            {navItems.map((item) => (
              <li key={item.path}>
                <Link 
                  to={item.path} 
                  className={`btn btn-ghost btn-sm ${
                    isActive(item.path) 
                      ? 'bg-primary text-primary-content shadow-lg' 
                      : 'hover:bg-base-200'
                  } transition-all duration-200`}
                >
                  <i className={item.icon}></i>
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </div>
        
        <div className="navbar-end">
          <div className="dropdown dropdown-end">
            <div tabIndex={0} role="button" className="btn btn-ghost btn-circle">
              <div className="indicator">
                <i className="fas fa-bell text-lg"></i>
                <span className="badge badge-xs badge-primary indicator-item pulse-glow"></span>
              </div>
            </div>
            <div 
              tabIndex={0} 
              className="mt-3 z-[1] card card-compact dropdown-content w-80 bg-base-100 shadow-xl border border-base-200"
            >
              <div className="card-body">
                <div className="flex items-center justify-between mb-3">
                  <span className="font-bold text-lg">System Status</span>
                  <div className="badge badge-success gap-1">
                    <div className="w-2 h-2 bg-success rounded-full animate-pulse"></div>
                    Online
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center p-2 bg-base-200 rounded-lg">
                    <span className="text-sm">AI Services</span>
                    <span className="text-xs text-success font-semibold">6/6 Active</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-base-200 rounded-lg">
                    <span className="text-sm">Blockchain</span>
                    <span className="text-xs text-success font-semibold">Synced</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-base-200 rounded-lg">
                    <span className="text-sm">Market Data</span>
                    <span className="text-xs text-success font-semibold">Live</span>
                  </div>
                </div>
                
                <div className="card-actions mt-4">
                  <Link to="/transactions" className="btn btn-primary btn-sm btn-block">
                    <i className="fas fa-chart-bar"></i>
                    View Dashboard
                  </Link>
                </div>
              </div>
            </div>
          </div>
          
          <div className="dropdown dropdown-end ml-2">
            <div tabIndex={0} role="button" className="btn btn-ghost btn-circle avatar">
              <div className="w-8 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
                <i className="fas fa-user text-white text-sm"></i>
              </div>
            </div>
            <ul 
              tabIndex={0} 
              className="menu menu-sm dropdown-content mt-3 z-[1] p-2 shadow-xl bg-base-100 rounded-box w-52 border border-base-200"
            >
              <li className="menu-title">
                <span>Account</span>
              </li>
              <li><a><i className="fas fa-user"></i> Profile</a></li>
              <li><a><i className="fas fa-cog"></i> Settings</a></li>
              <li><a><i className="fas fa-question-circle"></i> Help</a></li>
              <div className="divider my-1"></div>
              <li><a className="text-error"><i className="fas fa-sign-out-alt"></i> Logout</a></li>
            </ul>
          </div>
        </div>
      </div>
    </>
  );
};

export default Navbar;
