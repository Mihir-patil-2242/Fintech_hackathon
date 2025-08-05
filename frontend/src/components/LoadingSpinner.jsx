import React from 'react';

const LoadingSpinner = ({ size = 'md', message = 'Loading...' }) => {
  return (
    <div style={{ textAlign: 'center', padding: '2rem' }}>
      <div className="loading"></div>
      <p style={{ marginTop: '1rem', color: '#64748b' }}>{message}</p>
    </div>
  );
};

export default LoadingSpinner;
