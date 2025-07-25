// src/App.tsx
import React, { useState, useEffect } from 'react';
import { TherapySession } from './components/TherapySession';

// simple API service functions
const API_BASE = 'http://localhost:8000';

const checkHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch (error) {
    return false;
  }
};

const generateSessionId = (): string => {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [sessionStarted, setSessionStarted] = useState(false);

  // check API connection on app load
  useEffect(() => {
    const checkConnection = async () => {
      const isHealthy = await checkHealth();
      setIsConnected(isHealthy);
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  // start a new therapy session
  const startNewSession = () => {
    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    setSessionStarted(true);
  };

  // end current therapy session
  const endSession = () => {
    setSessionId(null);
    setSessionStarted(false);
  };

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#f8fafc',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      {/* header */}
      <header style={{
        backgroundColor: 'white',
        borderBottom: '1px solid #e2e8f0',
        padding: '1rem 0'
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          padding: '0 1rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <h1 style={{ margin: 0, color: '#1e293b' }}>AI Therapist</h1>

          <div style={{
            padding: '0.5rem 1rem',
            borderRadius: '0.5rem',
            backgroundColor: isConnected ? '#dcfce7' : '#fee2e2',
            color: isConnected ? '#166534' : '#991b1b',
            fontSize: '0.875rem'
          }}>
            {isConnected ? '● Connected' : '● Disconnected'}
          </div>
        </div>
      </header>

      {/* main content */}
      <main style={{
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '2rem 1rem'
      }}>
        {/* welcome section */}
        {!sessionStarted && (
          <div style={{ textAlign: 'center', padding: '3rem 0' }}>
            <h2 style={{
              fontSize: '2.5rem',
              fontWeight: 'bold',
              color: '#1e293b',
              marginBottom: '1.5rem'
            }}>
              Welcome to AI Therapist
            </h2>
            <p style={{
              fontSize: '1.125rem',
              color: '#64748b',
              marginBottom: '2rem',
              lineHeight: '1.6'
            }}>
              A safe space to explore your thoughts and emotions with AI-powered support.
            </p>

            <button
              onClick={startNewSession}
              disabled={!isConnected}
              style={{
                backgroundColor: isConnected ? '#4f46e5' : '#9ca3af',
                color: 'white',
                padding: '0.75rem 2rem',
                borderRadius: '0.5rem',
                border: 'none',
                fontSize: '1rem',
                fontWeight: '500',
                cursor: isConnected ? 'pointer' : 'not-allowed'
              }}
            >
              {isConnected ? 'Start New Session' : 'Connecting...'}
            </button>
          </div>
        )}

        {/* therapy session */}
        {sessionStarted && sessionId && (
          <TherapySession
            sessionId={sessionId}
            isConnected={isConnected}
            onEndSession={endSession}
          />
        )}
      </main>
    </div>
  );
}

export default App;