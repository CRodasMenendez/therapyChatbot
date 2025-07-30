// src/App.tsx
import React, { useState, useEffect } from 'react';
import { TherapySession } from './components/TherapySession';

const API_BASE = 'http://localhost:8000';

// Function to check if the backend API is healthy and responsive
const checkHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch (error) {
    return false;
  }
};

// Generate a unique session ID for each therapy session
const generateSessionId = (): string => {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [sessionStarted, setSessionStarted] = useState(false);

  // Effect to check backend connection status periodically
  useEffect(() => {
    const checkConnection = async () => {
      const isHealthy = await checkHealth();
      setIsConnected(isHealthy);
    };

    // Initial connection check
    checkConnection();
    // Check connection every 30 seconds
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  // Start a new therapy session with generated session ID
  const startNewSession = () => {
    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    setSessionStarted(true);
  };

  // End the current therapy session and return to landing page
  const endSession = () => {
    setSessionId(null);
    setSessionStarted(false);
  };

  return (
    <>
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
      }}>
        {/* Animated background elements for visual enhancement */}
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          overflow: 'hidden',
          zIndex: 0,
          pointerEvents: 'none'
        }}>
          <div style={{
            position: 'absolute',
            top: '10%',
            left: '10%',
            width: '300px',
            height: '300px',
            background: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '50%'
          }}></div>
          <div style={{
            position: 'absolute',
            top: '60%',
            right: '15%',
            width: '200px',
            height: '200px',
            background: 'rgba(255, 255, 255, 0.08)',
            borderRadius: '50%'
          }}></div>
        </div>

        {/* Header with app branding and connection status */}
        <header style={{
          position: 'relative',
          zIndex: 10,
          background: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
          padding: '1rem 0'
        }}>
          <div style={{
            maxWidth: '1200px',
            margin: '0 auto',
            padding: '0 2rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <div style={{
                width: '40px',
                height: '40px',
                background: 'linear-gradient(45deg, #FF6B6B, #4ECDC4)',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '20px',
                color: 'white'
              }}>
                AI
              </div>
              <h1 style={{
                margin: 0,
                color: 'white',
                fontSize: '1.5rem',
                fontWeight: '600',
                textShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                AI Therapist
              </h1>
            </div>

            {/* Connection status indicator */}
            <div style={{
              padding: '0.5rem 1rem',
              borderRadius: '25px',
              background: isConnected
                ? 'linear-gradient(45deg, #4CAF50, #45a049)'
                : 'linear-gradient(45deg, #f44336, #d32f2f)',
              color: 'white',
              fontSize: '0.875rem',
              fontWeight: '500',
              boxShadow: '0 4px 15px rgba(0,0,0,0.2)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              <div style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: 'white'
              }}></div>
              {isConnected ? 'Connected' : 'Disconnected'}
            </div>
          </div>
        </header>

        {/* Main content area */}
        <main style={{
          position: 'relative',
          zIndex: 10,
          maxWidth: '1200px',
          margin: '0 auto',
          padding: '1.5rem'
        }}>
          {/* Welcome section - shown when no session is active */}
          {!sessionStarted && (
            <div style={{
              textAlign: 'center',
              padding: '2.5rem 2rem',
              background: 'rgba(255, 255, 255, 0.1)',
              backdropFilter: 'blur(20px)',
              borderRadius: '20px',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              boxShadow: '0 20px 40px rgba(0,0,0,0.1)'
            }}>
              <div style={{ maxWidth: '800px', margin: '0 auto' }}>
                {/* App icon/logo */}
                <div style={{
                  width: '80px',
                  height: '80px',
                  background: 'linear-gradient(45deg, #FF6B6B, #4ECDC4)',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '1.5rem',
                  margin: '0 auto 1.5rem',
                  boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
                  color: 'white',
                  fontWeight: 'bold'
                }}>
                  CHAT
                </div>

                {/* Main welcome heading */}
                <h2 style={{
                  fontSize: '2.5rem',
                  fontWeight: '700',
                  color: 'white',
                  marginBottom: '0.75rem',
                  textShadow: '0 4px 8px rgba(0,0,0,0.2)'
                }}>
                  Welcome to AI Therapist
                </h2>

                {/* App description */}
                <p style={{
                  fontSize: '1.1rem',
                  color: 'rgba(255, 255, 255, 0.9)',
                  marginBottom: '2rem',
                  lineHeight: '1.6',
                  textShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  Your personal AI companion for emotional support and mental wellness.
                  Experience empathetic conversations powered by advanced emotion recognition
                  and therapeutic response generation.
                </p>

                {/* Feature boxes - now only showing 2 features */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                  gap: '1.25rem',
                  marginBottom: '2rem'
                }}>
                  {/* Voice & Text Enabled feature box */}
                  <div style={{
                    background: 'rgba(255, 255, 255, 0.15)',
                    backdropFilter: 'blur(10px)',
                    padding: '1.5rem',
                    borderRadius: '15px',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    transition: 'all 0.3s ease',
                    cursor: 'default'
                  }}>
                    <div style={{
                      width: '50px',
                      height: '50px',
                      background: 'linear-gradient(45deg, #667eea, #764ba2)',
                      borderRadius: '15px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto 0.75rem',
                      fontSize: '1rem',
                      color: 'white',
                      fontWeight: 'bold'
                    }}>
                      MIC
                    </div>
                    <h3 style={{
                      fontWeight: '600',
                      color: 'white',
                      marginBottom: '0.5rem',
                      fontSize: '1.05rem'
                    }}>
                      Voice & Text Enabled
                    </h3>
                    <p style={{
                      color: 'rgba(255, 255, 255, 0.8)',
                      fontSize: '0.85rem',
                      lineHeight: '1.4',
                      margin: 0
                    }}>
                      Express yourself naturally through speech or text. Our AI understands both forms of communication.
                    </p>
                  </div>

                  {/* Emotion Recognition feature box */}
                  <div style={{
                    background: 'rgba(255, 255, 255, 0.15)',
                    backdropFilter: 'blur(10px)',
                    padding: '1.5rem',
                    borderRadius: '15px',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    transition: 'all 0.3s ease',
                    cursor: 'default'
                  }}>
                    <div style={{
                      width: '50px',
                      height: '50px',
                      background: 'linear-gradient(45deg, #FF6B6B, #4ECDC4)',
                      borderRadius: '15px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto 0.75rem',
                      fontSize: '1rem',
                      color: 'white',
                      fontWeight: 'bold'
                    }}>
                      AI
                    </div>
                    <h3 style={{
                      fontWeight: '600',
                      color: 'white',
                      marginBottom: '0.5rem',
                      fontSize: '1.05rem'
                    }}>
                      Emotion Recognition
                    </h3>
                    <p style={{
                      color: 'rgba(255, 255, 255, 0.8)',
                      fontSize: '0.85rem',
                      lineHeight: '1.4',
                      margin: 0
                    }}>
                      Advanced AI detects 32 different emotions and adapts responses to your emotional state.
                    </p>
                  </div>
                </div>

                {/* Start session button */}
                <button
                  onClick={startNewSession}
                  disabled={!isConnected}
                  style={{
                    background: isConnected
                      ? 'linear-gradient(45deg, #FF6B6B, #4ECDC4)'
                      : 'linear-gradient(45deg, #666, #999)',
                    color: 'white',
                    padding: '1rem 2.5rem',
                    borderRadius: '50px',
                    border: 'none',
                    fontSize: '1.1rem',
                    fontWeight: '600',
                    cursor: isConnected ? 'pointer' : 'not-allowed',
                    transition: 'all 0.3s ease',
                    boxShadow: '0 10px 25px rgba(0,0,0,0.2)',
                    textTransform: 'uppercase',
                    letterSpacing: '1px'
                  }}
                >
                  {isConnected ? 'Start Your Journey' : 'Connecting...'}
                </button>

                {/* Connection status message when not connected */}
                {!isConnected && (
                  <p style={{
                    color: 'rgba(255, 255, 255, 0.7)',
                    fontSize: '0.9rem',
                    marginTop: '1rem'
                  }}>
                    Please wait while we establish connection to the AI service...
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Therapy session component - shown when session is active */}
          {sessionStarted && sessionId && (
            <TherapySession
              sessionId={sessionId}
              isConnected={isConnected}
              onEndSession={endSession}
            />
          )}
        </main>
      </div>
    </>
  );
}

export default App;