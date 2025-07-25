// src/components/TherapySession.tsx
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

    // add a new message to the conversation
    const addMessage = (message: Omit<Message, 'id'>) => {
        const newMessage: Message = {
            ...message,
            id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        };
        setMessages(prev => [...prev, newMessage]);
    };

    // connect to WebSocket
    const connectWebSocket = () => {
        if (!isConnected || wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        setWsStatus('connecting');

        try {
            const ws = new WebSocket(`ws://localhost:8000/ws/therapy/${sessionId}`);
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

    // handle audio recording completion
    const handleAudioSubmit = async (audioBlob: Blob) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected');
            return;
        }

        try {
            // convert audio to base64
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

    // handle text message submission
    const handleTextSubmit = (e: React.FormEvent) => {
        e.preventDefault();

        if (!textInput.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            return;
        }

        try {
            const message = {
                type: 'text_message',
                content: textInput.trim(),
                timestamp: new Date().toISOString(),
            };

            wsRef.current.send(JSON.stringify(message));
            setTextInput('');
        } catch (error) {
            console.error('Error sending text:', error);
        }
    };

    // connect WebSocket when component mounts
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

    // scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // add welcome message on session start
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
        <div style={{ maxWidth: '900px', margin: '0 auto' }}>
            {/* session header */}
            <div style={{
                backgroundColor: 'white',
                borderRadius: '0.5rem',
                border: '1px solid #e2e8f0',
                padding: '1.5rem',
                marginBottom: '1.5rem'
            }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1e293b', margin: '0 0 0.5rem 0' }}>
                            Therapy Session
                        </h2>
                        <p style={{ color: '#64748b', margin: 0 }}>
                            Connection: <span style={{
                                fontWeight: '500',
                                color: wsStatus === 'connected' ? '#059669' :
                                    wsStatus === 'connecting' ? '#d97706' : '#dc2626'
                            }}>
                                {wsStatus === 'connected' ? 'Connected' :
                                    wsStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
                            </span>
                        </p>
                    </div>

                    {/* current emotion indicator */}
                    {currentEmotion && (
                        <div style={{
                            padding: '0.5rem 1rem',
                            backgroundColor: '#f1f5f9',
                            borderRadius: '0.5rem',
                            border: '1px solid #e2e8f0'
                        }}>
                            <div style={{ fontSize: '0.875rem', color: '#64748b' }}>Current Emotion</div>
                            <div style={{ fontWeight: '600', color: '#1e293b' }}>
                                {currentEmotion} ({Math.round(emotionConfidence * 100)}%)
                            </div>
                        </div>
                    )}

                    {/* end session button */}
                    <button
                        onClick={onEndSession}
                        style={{
                            backgroundColor: '#dc2626',
                            color: 'white',
                            padding: '0.5rem 1rem',
                            borderRadius: '0.375rem',
                            border: 'none',
                            fontSize: '0.875rem',
                            cursor: 'pointer'
                        }}
                    >
                        End Session
                    </button>
                </div>
            </div>

            {/* chat messages */}
            <div style={{
                backgroundColor: 'white',
                borderRadius: '0.5rem',
                border: '1px solid #e2e8f0',
                marginBottom: '1.5rem'
            }}>
                <div style={{
                    height: '400px',
                    overflowY: 'auto',
                    padding: '1rem'
                }}>
                    {messages.map((message) => (
                        <div
                            key={message.id}
                            style={{
                                marginBottom: '1rem',
                                padding: '0.75rem',
                                borderRadius: '0.5rem',
                                backgroundColor:
                                    message.type === 'user' ? '#eff6ff' :
                                        message.type === 'therapist' ? '#f3e8ff' : '#f8fafc'
                            }}
                        >
                            <div style={{
                                fontWeight: '600',
                                color: '#1e293b',
                                marginBottom: '0.25rem',
                                fontSize: '0.875rem'
                            }}>
                                {message.type === 'user' ? 'You' :
                                    message.type === 'therapist' ? 'Therapist' : 'System'}
                            </div>
                            <p style={{ margin: '0.25rem 0', color: '#374151' }}>
                                {message.content}
                            </p>
                            {message.emotion && (
                                <div style={{
                                    fontSize: '0.75rem',
                                    color: '#6b7280',
                                    marginTop: '0.25rem'
                                }}>
                                    Detected emotion: {message.emotion}
                                    {message.confidence && ` (${Math.round(message.confidence * 100)}%)`}
                                </div>
                            )}
                        </div>
                    ))}

                    {/* typing indicator */}
                    {isTherapistTyping && (
                        <div style={{
                            padding: '0.75rem',
                            borderRadius: '0.5rem',
                            backgroundColor: '#f3e8ff',
                            color: '#6b7280',
                            fontStyle: 'italic'
                        }}>
                            Therapist is typing...
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* input controls */}
            <div style={{
                backgroundColor: 'white',
                borderRadius: '0.5rem',
                border: '1px solid #e2e8f0',
                padding: '1.5rem'
            }}>
                {/* input mode toggle */}
                <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    marginBottom: '1.5rem'
                }}>
                    <div style={{
                        backgroundColor: '#f8fafc',
                        borderRadius: '0.5rem',
                        padding: '0.25rem',
                        display: 'flex'
                    }}>
                        <button
                            onClick={() => setInputMode('voice')}
                            style={{
                                padding: '0.5rem 1rem',
                                borderRadius: '0.375rem',
                                border: 'none',
                                fontWeight: '500',
                                cursor: 'pointer',
                                backgroundColor: inputMode === 'voice' ? '#4f46e5' : 'transparent',
                                color: inputMode === 'voice' ? 'white' : '#64748b'
                            }}
                        >
                            Voice
                        </button>
                        <button
                            onClick={() => setInputMode('text')}
                            style={{
                                padding: '0.5rem 1rem',
                                borderRadius: '0.375rem',
                                border: 'none',
                                fontWeight: '500',
                                cursor: 'pointer',
                                backgroundColor: inputMode === 'text' ? '#4f46e5' : 'transparent',
                                color: inputMode === 'text' ? 'white' : '#64748b'
                            }}
                        >
                            Text
                        </button>
                    </div>
                </div>

                {/* voice input */}
                {inputMode === 'voice' && (
                    <AudioRecorder
                        onAudioSubmit={handleAudioSubmit}
                        disabled={wsStatus !== 'connected'}
                    />
                )}

                {/* text input */}
                {inputMode === 'text' && (
                    <form onSubmit={handleTextSubmit} style={{ display: 'flex', gap: '1rem' }}>
                        <input
                            type="text"
                            value={textInput}
                            onChange={(e) => setTextInput(e.target.value)}
                            placeholder="Type your message here..."
                            disabled={wsStatus !== 'connected'}
                            style={{
                                flex: 1,
                                padding: '0.75rem 1rem',
                                border: '1px solid #d1d5db',
                                borderRadius: '0.5rem',
                                fontSize: '1rem',
                                outline: 'none',
                                opacity: wsStatus !== 'connected' ? 0.5 : 1
                            }}
                        />
                        <button
                            type="submit"
                            disabled={!textInput.trim() || wsStatus !== 'connected'}
                            style={{
                                padding: '0.75rem 1.5rem',
                                backgroundColor: (!textInput.trim() || wsStatus !== 'connected') ? '#9ca3af' : '#4f46e5',
                                color: 'white',
                                borderRadius: '0.5rem',
                                border: 'none',
                                fontWeight: '500',
                                cursor: (!textInput.trim() || wsStatus !== 'connected') ? 'not-allowed' : 'pointer'
                            }}
                        >
                            Send
                        </button>
                    </form>
                )}

                {/* connection status message */}
                {wsStatus !== 'connected' && (
                    <div style={{
                        marginTop: '1rem',
                        padding: '0.75rem',
                        backgroundColor: '#fef3c7',
                        border: '1px solid #f59e0b',
                        borderRadius: '0.5rem',
                        textAlign: 'center'
                    }}>
                        <p style={{
                            color: '#92400e',
                            fontSize: '0.875rem',
                            margin: 0
                        }}>
                            {wsStatus === 'connecting'
                                ? 'Connecting to therapy session...'
                                : 'Disconnected. Attempting to reconnect...'}
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};