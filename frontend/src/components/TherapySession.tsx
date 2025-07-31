// src/components/TherapySession.tsx
import config from '../config';
import React, { useState, useEffect, useRef } from 'react';
import { AudioRecorder } from './AudioRecorder';


interface TherapySessionProps {
    sessionId: string;
    isConnected: boolean;
    onEndSession: () => void;
}

interface Message {
    id: string;
    type: 'user' | 'therapist' | 'system';
    content: string;
    emotion?: string;
    confidence?: number;
    timestamp: Date;
}

export const TherapySession: React.FC<TherapySessionProps> = ({
    sessionId,
    isConnected,
    onEndSession
}) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isTherapistTyping, setIsTherapistTyping] = useState(false);
    const [currentEmotion, setCurrentEmotion] = useState<string | null>(null);
    const [emotionConfidence, setEmotionConfidence] = useState<number>(0);
    const [inputMode, setInputMode] = useState<'voice' | 'text'>('voice');
    const [textInput, setTextInput] = useState('');
    const [wsStatus, setWsStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');

    const wsRef = useRef<WebSocket | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Helper function to get emotion color based on emotion type
    const getEmotionColor = (emotion: string): string => {
        const colors: { [key: string]: string } = {
            'joyful': '#4CAF50', 'excited': '#FF9800', 'grateful': '#8BC34A',
            'hopeful': '#2196F3', 'confident': '#3F51B5', 'content': '#4CAF50',
            'angry': '#F44336', 'sad': '#607D8B', 'anxious': '#FF5722',
            'afraid': '#795548', 'lonely': '#9E9E9E', 'frustrated': '#E91E63'
        };
        return colors[emotion] || '#9C27B0';
    };

    // Function to add a new message to the chat
    const addMessage = (message: Omit<Message, 'id'>) => {
        const newMessage: Message = {
            ...message,
            id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        };
        setMessages(prev => [...prev, newMessage]);
    };

    // WebSocket connection management
    const connectWebSocket = () => {
        if (!isConnected || wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        setWsStatus('connecting');

        try {
            const ws = new WebSocket(`${config.WS_BASE}/ws/therapy/${sessionId}`);
            wsRef.current = ws;

            ws.onopen = () => {
                setWsStatus('connected');
            };

            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);

                    switch (message.type) {
                        case 'therapist_message':
                            setIsTherapistTyping(false);
                            addMessage({
                                type: 'therapist',
                                content: message.content,
                                timestamp: new Date()
                            });
                            break;

                        case 'transcription':
                            addMessage({
                                type: 'user',
                                content: message.content,
                                emotion: message.emotion,
                                confidence: message.confidence,
                                timestamp: new Date()
                            });

                            if (message.emotion) {
                                setCurrentEmotion(message.emotion);
                                setEmotionConfidence(message.confidence || 0);
                            }

                            setIsTherapistTyping(true);
                            break;

                        case 'error':
                            console.error('WebSocket error:', message.content);
                            setIsTherapistTyping(false);
                            break;
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            ws.onclose = () => {
                setWsStatus('disconnected');
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setWsStatus('disconnected');
            };

        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            setWsStatus('disconnected');
        }
    };

    // Handle audio submission from the AudioRecorder component
    const handleAudioSubmit = async (audioBlob: Blob) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected');
            return;
        }

        try {
            const reader = new FileReader();
            reader.onload = () => {
                const arrayBuffer = reader.result as ArrayBuffer;
                const uint8Array = new Uint8Array(arrayBuffer);
                let binary = '';
                for (let i = 0; i < uint8Array.length; i++) {
                    binary += String.fromCharCode(uint8Array[i]);
                }
                const base64String = btoa(binary);

                const message = {
                    type: 'audio_data',
                    audio: base64String,
                    timestamp: new Date().toISOString(),
                };

                wsRef.current?.send(JSON.stringify(message));
            };
            reader.readAsArrayBuffer(audioBlob);
        } catch (error) {
            console.error('Error sending audio:', error);
        }
    };

    // Handle text message submission
    const handleTextSubmit = (e: React.FormEvent) => {
        e.preventDefault();

        if (!textInput.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            return;
        }

        try {
            const messageContent = textInput.trim();

            // Immediately add the user's message to the chat display
            addMessage({
                type: 'user',
                content: messageContent,
                timestamp: new Date()
            });

            // Send the message to the backend
            const message = {
                type: 'text_message',
                content: messageContent,
                timestamp: new Date().toISOString(),
            };

            wsRef.current.send(JSON.stringify(message));
            setTextInput('');

            // Set therapist typing indicator since we expect a response
            setIsTherapistTyping(true);
        } catch (error) {
            console.error('Error sending text:', error);
        }
    };

    // Effect to handle WebSocket connection when component mounts or connection status changes
    useEffect(() => {
        if (isConnected) {
            connectWebSocket();
        }

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [isConnected]);

    // Effect to auto-scroll to bottom when new messages arrive
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Effect to add initial system message when component mounts
    useEffect(() => {
        if (messages.length === 0) {
            addMessage({
                type: 'system',
                content: 'Session started. You can speak or type to begin your conversation.',
                timestamp: new Date()
            });
        }
    }, []);

    return (
        <div style={{
            maxWidth: '900px',
            margin: '0 auto',
            background: 'rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(20px)',
            borderRadius: '20px',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            overflow: 'hidden',
            boxShadow: '0 20px 40px rgba(0,0,0,0.1)'
        }}>
            {/* Session header */}
            <div style={{
                background: 'rgba(255, 255, 255, 0.15)',
                padding: '2rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
            }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <h2 style={{
                            fontSize: '1.8rem',
                            fontWeight: '600',
                            color: 'white',
                            margin: '0 0 0.5rem 0',
                            textShadow: '0 2px 4px rgba(0,0,0,0.2)'
                        }}>
                            Therapy Session
                        </h2>

                        {/* Therapist thinking indicator - only show when therapist is typing */}
                        {isTherapistTyping && (
                            <div style={{
                                padding: '1rem 1.25rem',
                                borderRadius: '20px 20px 20px 5px',
                                background: 'rgba(78, 205, 196, 0.3)',
                                backdropFilter: 'blur(10px)',
                                border: '1px solid rgba(255, 255, 255, 0.2)',
                                color: 'white',
                                fontStyle: 'italic',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.5rem'
                            }}>
                                <div style={{
                                    display: 'flex',
                                    gap: '3px'
                                }}>
                                    <div style={{
                                        width: '8px',
                                        height: '8px',
                                        borderRadius: '50%',
                                        backgroundColor: 'white'
                                    }}></div>
                                    <div style={{
                                        width: '8px',
                                        height: '8px',
                                        borderRadius: '50%',
                                        backgroundColor: 'white'
                                    }}></div>
                                    <div style={{
                                        width: '8px',
                                        height: '8px',
                                        borderRadius: '50%',
                                        backgroundColor: 'white'
                                    }}></div>
                                </div>
                                AI Therapist is thinking...
                            </div>
                        )}
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        {/* Connection status indicator */}
                        <div style={{
                            padding: '0.25rem 0.75rem',
                            borderRadius: '15px',
                            background: wsStatus === 'connected'
                                ? 'linear-gradient(45deg, #4CAF50, #45a049)'
                                : wsStatus === 'connecting'
                                    ? 'linear-gradient(45deg, #FF9800, #F57C00)'
                                    : 'linear-gradient(45deg, #f44336, #d32f2f)',
                            color: 'white',
                            fontSize: '0.8rem',
                            fontWeight: '500'
                        }}>
                            {wsStatus === 'connected' ? 'Connected' :
                                wsStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
                        </div>

                        {/* Current emotion indicator */}
                        {currentEmotion && (
                            <div style={{
                                padding: '1rem',
                                background: 'rgba(255, 255, 255, 0.2)',
                                borderRadius: '15px',
                                border: '1px solid rgba(255, 255, 255, 0.3)',
                                textAlign: 'center',
                                minWidth: '150px'
                            }}>
                                <div style={{
                                    fontSize: '0.8rem',
                                    color: 'rgba(255, 255, 255, 0.8)',
                                    marginBottom: '0.25rem'
                                }}>
                                    Current Emotion
                                </div>
                                <div style={{
                                    fontWeight: '600',
                                    color: 'white',
                                    fontSize: '1rem',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: '0.5rem'
                                }}>
                                    <div style={{
                                        width: '12px',
                                        height: '12px',
                                        borderRadius: '50%',
                                        backgroundColor: getEmotionColor(currentEmotion)
                                    }}></div>
                                    {currentEmotion} ({Math.round(emotionConfidence * 100)}%)
                                </div>
                            </div>
                        )}

                        {/* End session button */}
                        <button
                            onClick={onEndSession}
                            style={{
                                background: 'linear-gradient(45deg, #f44336, #d32f2f)',
                                color: 'white',
                                padding: '0.75rem 1.5rem',
                                borderRadius: '25px',
                                border: 'none',
                                fontSize: '0.9rem',
                                fontWeight: '500',
                                cursor: 'pointer',
                                transition: 'all 0.3s ease',
                                boxShadow: '0 4px 15px rgba(244, 67, 54, 0.3)'
                            }}
                        >
                            End Session
                        </button>
                    </div>
                </div>
            </div>

            {/* Chat messages */}
            <div style={{
                height: '450px',
                overflowY: 'auto',
                padding: '1.5rem',
                background: 'rgba(255, 255, 255, 0.05)'
            }}>
                {messages.map((message) => (
                    <div
                        key={message.id}
                        style={{
                            marginBottom: '1.5rem',
                            display: 'flex',
                            flexDirection: message.type === 'user' ? 'row-reverse' : 'row',
                            alignItems: 'flex-start',
                            gap: '1rem'
                        }}
                    >
                        {/* Avatar */}
                        <div style={{
                            width: '40px',
                            height: '40px',
                            borderRadius: '50%',
                            background: message.type === 'user'
                                ? 'linear-gradient(45deg, #667eea, #764ba2)'
                                : message.type === 'therapist'
                                    ? 'linear-gradient(45deg, #4ECDC4, #44A08D)'
                                    : 'linear-gradient(45deg, #95a5a6, #7f8c8d)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '1.2rem',
                            flexShrink: 0,
                            color: 'white',
                            fontWeight: 'bold'
                        }}>
                            {message.type === 'user' ? 'U' : message.type === 'therapist' ? 'AI' : 'SYS'}
                        </div>

                        {/* Message bubble */}
                        <div style={{
                            maxWidth: '70%',
                            padding: '1rem 1.25rem',
                            borderRadius: message.type === 'user' ? '20px 20px 5px 20px' : '20px 20px 20px 5px',
                            background: message.type === 'user'
                                ? 'linear-gradient(45deg, rgba(102, 126, 234, 0.9), rgba(118, 75, 162, 0.9))'
                                : message.type === 'therapist'
                                    ? 'linear-gradient(45deg, rgba(78, 205, 196, 0.9), rgba(68, 160, 141, 0.9))'
                                    : 'rgba(255, 255, 255, 0.2)',
                            color: 'white',
                            backdropFilter: 'blur(10px)',
                            border: '1px solid rgba(255, 255, 255, 0.2)',
                            boxShadow: '0 8px 25px rgba(0,0,0,0.15)'
                        }}>
                            <div style={{
                                fontSize: '0.75rem',
                                opacity: 0.8,
                                marginBottom: '0.5rem',
                                fontWeight: '500'
                            }}>
                                {message.type === 'user' ? 'You' :
                                    message.type === 'therapist' ? 'AI Therapist' : 'System'}
                            </div>

                            <p style={{
                                margin: 0,
                                lineHeight: '1.5',
                                fontSize: '0.95rem'
                            }}>
                                {message.content}
                            </p>

                            {/* Emotion indicator for user messages */}
                            {message.emotion && (
                                <div style={{
                                    marginTop: '0.75rem',
                                    padding: '0.4rem 0.8rem',
                                    background: 'rgba(255, 255, 255, 0.2)',
                                    borderRadius: '12px',
                                    fontSize: '0.7rem',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '0.5rem'
                                }}>
                                    <div style={{
                                        width: '8px',
                                        height: '8px',
                                        borderRadius: '50%',
                                        backgroundColor: getEmotionColor(message.emotion)
                                    }}></div>
                                    {message.emotion}
                                    {message.confidence && ` (${Math.round(message.confidence * 100)}%)`}
                                </div>
                            )}
                        </div>
                    </div>
                ))}

                {/* Typing indicator when therapist is responding */}
                {isTherapistTyping && (
                    <div style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '1rem',
                        marginBottom: '1.5rem'
                    }}>
                        <div style={{
                            width: '40px',
                            height: '40px',
                            borderRadius: '50%',
                            background: 'linear-gradient(45deg, #4ECDC4, #44A08D)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '1.2rem',
                            color: 'white',
                            fontWeight: 'bold'
                        }}>
                            AI
                        </div>
                        <div style={{
                            padding: '1rem 1.25rem',
                            borderRadius: '20px 20px 20px 5px',
                            background: 'rgba(78, 205, 196, 0.3)',
                            backdropFilter: 'blur(10px)',
                            border: '1px solid rgba(255, 255, 255, 0.2)',
                            color: 'white',
                            fontStyle: 'italic',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.5rem'
                        }}>
                            <div style={{
                                display: 'flex',
                                gap: '3px'
                            }}>
                                <div style={{
                                    width: '8px',
                                    height: '8px',
                                    borderRadius: '50%',
                                    backgroundColor: 'white'
                                }}></div>
                                <div style={{
                                    width: '8px',
                                    height: '8px',
                                    borderRadius: '50%',
                                    backgroundColor: 'white'
                                }}></div>
                                <div style={{
                                    width: '8px',
                                    height: '8px',
                                    borderRadius: '50%',
                                    backgroundColor: 'white'
                                }}></div>
                            </div>
                            Typing...
                        </div>
                    </div>
                )}

                {/* Scroll anchor */}
                <div ref={messagesEndRef} />
            </div>

            {/* Input controls */}
            <div style={{
                padding: '2rem',
                background: 'rgba(255, 255, 255, 0.1)',
                borderTop: '1px solid rgba(255, 255, 255, 0.1)'
            }}>
                {/* Input mode toggle */}
                <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    marginBottom: '1.5rem'
                }}>
                    <div style={{
                        background: 'rgba(255, 255, 255, 0.2)',
                        borderRadius: '25px',
                        padding: '0.4rem',
                        display: 'flex',
                        border: '1px solid rgba(255, 255, 255, 0.3)'
                    }}>
                        <button
                            onClick={() => setInputMode('voice')}
                            style={{
                                padding: '0.6rem 1.5rem',
                                borderRadius: '20px',
                                border: 'none',
                                fontWeight: '500',
                                cursor: 'pointer',
                                transition: 'all 0.3s ease',
                                background: inputMode === 'voice'
                                    ? 'linear-gradient(45deg, #FF6B6B, #4ECDC4)'
                                    : 'transparent',
                                color: 'white',
                                fontSize: '0.9rem'
                            }}
                        >
                            Voice
                        </button>
                        <button
                            onClick={() => setInputMode('text')}
                            style={{
                                padding: '0.6rem 1.5rem',
                                borderRadius: '20px',
                                border: 'none',
                                fontWeight: '500',
                                cursor: 'pointer',
                                transition: 'all 0.3s ease',
                                background: inputMode === 'text'
                                    ? 'linear-gradient(45deg, #FF6B6B, #4ECDC4)'
                                    : 'transparent',
                                color: 'white',
                                fontSize: '0.9rem'
                            }}
                        >
                            Text
                        </button>
                    </div>
                </div>

                {/* Voice input */}
                {inputMode === 'voice' && (
                    <AudioRecorder
                        onAudioSubmit={handleAudioSubmit}
                        disabled={wsStatus !== 'connected'}
                    />
                )}

                {/* Text input */}
                {inputMode === 'text' && (
                    <form onSubmit={handleTextSubmit} style={{ display: 'flex', gap: '1rem' }}>
                        <input
                            type="text"
                            value={textInput}
                            onChange={(e) => setTextInput(e.target.value)}
                            placeholder="Share your thoughts..."
                            disabled={wsStatus !== 'connected'}
                            style={{
                                flex: 1,
                                padding: '1rem 1.5rem',
                                border: '1px solid rgba(255, 255, 255, 0.3)',
                                borderRadius: '25px',
                                fontSize: '1rem',
                                outline: 'none',
                                background: 'rgba(255, 255, 255, 0.2)',
                                backdropFilter: 'blur(10px)',
                                color: 'white',
                                opacity: wsStatus !== 'connected' ? 0.5 : 1
                            }}
                        />
                        <button
                            type="submit"
                            disabled={!textInput.trim() || wsStatus !== 'connected'}
                            style={{
                                padding: '1rem 2rem',
                                background: (!textInput.trim() || wsStatus !== 'connected')
                                    ? 'rgba(255, 255, 255, 0.2)'
                                    : 'linear-gradient(45deg, #4ECDC4, #44A08D)',
                                color: 'white',
                                borderRadius: '25px',
                                border: 'none',
                                fontWeight: '500',
                                cursor: (!textInput.trim() || wsStatus !== 'connected') ? 'not-allowed' : 'pointer',
                                transition: 'all 0.3s ease'
                            }}
                        >
                            Send
                        </button>
                    </form>
                )}

                {/* Connection status message */}
                {wsStatus !== 'connected' && (
                    <div style={{
                        marginTop: '1rem',
                        padding: '1rem',
                        background: 'rgba(255, 152, 0, 0.2)',
                        border: '1px solid rgba(255, 152, 0, 0.3)',
                        borderRadius: '15px',
                        textAlign: 'center',
                        backdropFilter: 'blur(10px)'
                    }}>
                        <p style={{
                            color: 'rgba(255, 255, 255, 0.9)',
                            fontSize: '0.9rem',
                            margin: 0
                        }}>
                            {wsStatus === 'connecting'
                                ? 'Connecting to therapy session...'
                                : 'Connection lost. Attempting to reconnect...'}
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};