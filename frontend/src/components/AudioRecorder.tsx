// src/components/AudioRecorder.tsx
import React, { useState, useRef } from 'react';

interface AudioRecorderProps {
    onAudioSubmit: (audioBlob: Blob) => void;
    disabled?: boolean;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({
    onAudioSubmit,
    disabled = false
}) => {
    const [isRecording, setIsRecording] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [recordingTime, setRecordingTime] = useState(0);
    const [error, setError] = useState<string | null>(null);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const chunksRef = useRef<Blob[]>([]);

    const startTimer = () => {
        timerRef.current = setInterval(() => {
            setRecordingTime(prev => prev + 1);
        }, 1000);
    };

    const stopTimer = () => {
        if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
        }
    };

    const resetTimer = () => {
        stopTimer();
        setRecordingTime(0);
    };

    const startRecording = async () => {
        try {
            setError(null);
            setAudioBlob(null);
            chunksRef.current = [];

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 44100,
                }
            });

            streamRef.current = stream;

            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            mediaRecorderRef.current = mediaRecorder;

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    chunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
                setAudioBlob(blob);

                if (streamRef.current) {
                    streamRef.current.getTracks().forEach(track => track.stop());
                    streamRef.current = null;
                }
            };

            mediaRecorder.onerror = (event) => {
                console.error('MediaRecorder error:', event);
                setError('Recording error occurred');
                setIsRecording(false);
                resetTimer();
            };

            mediaRecorder.start(100);
            setIsRecording(true);
            setIsPaused(false);
            resetTimer();
            startTimer();

        } catch (err) {
            console.error('Error starting recording:', err);

            if (err instanceof Error) {
                if (err.name === 'NotAllowedError') {
                    setError('Microphone access denied. Please allow microphone access and try again.');
                } else if (err.name === 'NotFoundError') {
                    setError('No microphone found. Please check your microphone connection.');
                } else {
                    setError(`Failed to start recording: ${err.message}`);
                }
            } else {
                setError('Failed to start recording. Please check your microphone.');
            }
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            setIsPaused(false);
            stopTimer();
        }
    };

    const pauseRecording = () => {
        if (mediaRecorderRef.current && isRecording && !isPaused) {
            mediaRecorderRef.current.pause();
            setIsPaused(true);
            stopTimer();
        }
    };

    const resumeRecording = () => {
        if (mediaRecorderRef.current && isRecording && isPaused) {
            mediaRecorderRef.current.resume();
            setIsPaused(false);
            startTimer();
        }
    };

    const clearRecording = () => {
        setAudioBlob(null);
        setError(null);
        chunksRef.current = [];
        resetTimer();

        if (isRecording) {
            stopRecording();
        }
    };

    const handleSubmit = () => {
        if (audioBlob) {
            onAudioSubmit(audioBlob);
            clearRecording();
        }
    };

    const formatRecordingTime = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <>
            <div style={{ textAlign: 'center' }}>
                {/* error display */}
                {error && (
                    <div style={{
                        padding: '1rem',
                        background: 'rgba(244, 67, 54, 0.2)',
                        border: '1px solid rgba(244, 67, 54, 0.3)',
                        borderRadius: '15px',
                        marginBottom: '1.5rem',
                        backdropFilter: 'blur(10px)'
                    }}>
                        <p style={{ color: 'rgba(255, 255, 255, 0.9)', fontSize: '0.9rem', margin: 0 }}>
                            ⚠ {error}
                        </p>
                    </div>
                )}

                {/* main recording controls */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '1.5rem',
                    marginBottom: '1.5rem'
                }}>
                    {/* record/stop button */}
                    {!isRecording ? (
                        <button
                            onClick={startRecording}
                            disabled={disabled}
                            style={{
                                width: '80px',
                                height: '80px',
                                background: disabled
                                    ? 'rgba(255, 255, 255, 0.2)'
                                    : 'linear-gradient(45deg, #FF6B6B, #4ECDC4)',
                                color: 'white',
                                borderRadius: '50%',
                                border: 'none',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                cursor: disabled ? 'not-allowed' : 'pointer',
                                fontSize: '1.5rem',
                                transition: 'all 0.3s ease',
                                boxShadow: '0 10px 25px rgba(0,0,0,0.2)',
                                backdropFilter: 'blur(10px)',
                                fontWeight: 'bold'
                            }}
                        >
                            MIC
                        </button>
                    ) : (
                        <button
                            onClick={stopRecording}
                            style={{
                                width: '80px',
                                height: '80px',
                                background: 'linear-gradient(45deg, #f44336, #d32f2f)',
                                color: 'white',
                                borderRadius: '50%',
                                border: 'none',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                cursor: 'pointer',
                                fontSize: '1.2rem',
                                transition: 'all 0.3s ease',
                                boxShadow: '0 0 0 0 rgba(244, 67, 54, 0.7)',
                                fontWeight: 'bold'
                            }}
                        >
                            STOP
                        </button>
                    )}

                    {/* pause/resume button */}
                    {isRecording && (
                        <button
                            onClick={isPaused ? resumeRecording : pauseRecording}
                            style={{
                                width: '60px',
                                height: '60px',
                                background: 'linear-gradient(45deg, #2196F3, #1976D2)',
                                color: 'white',
                                borderRadius: '50%',
                                border: 'none',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                cursor: 'pointer',
                                fontSize: '0.8rem',
                                transition: 'all 0.3s ease',
                                boxShadow: '0 8px 20px rgba(33, 150, 243, 0.3)',
                                fontWeight: 'bold'
                            }}
                        >
                            {isPaused ? 'PLAY' : 'PAUSE'}
                        </button>
                    )}
                </div>

                {/* recording timer */}
                {isRecording && (
                    <div style={{ marginBottom: '1.5rem' }}>
                        <div style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.75rem',
                            background: 'rgba(255, 255, 255, 0.2)',
                            backdropFilter: 'blur(10px)',
                            padding: '1rem 1.5rem',
                            borderRadius: '25px',
                            border: '1px solid rgba(255, 255, 255, 0.3)'
                        }}>
                            <div style={{
                                width: '12px',
                                height: '12px',
                                backgroundColor: '#f44336',
                                borderRadius: '50%'
                            }}></div>
                            <span style={{
                                fontFamily: 'monospace',
                                color: 'white',
                                fontSize: '1.1rem',
                                fontWeight: '600'
                            }}>
                                {formatRecordingTime(recordingTime)}
                            </span>
                            {isPaused && (
                                <span style={{
                                    color: 'rgba(255, 255, 255, 0.8)',
                                    fontSize: '0.9rem',
                                    fontStyle: 'italic'
                                }}>
                                    (Paused)
                                </span>
                            )}
                        </div>
                    </div>
                )}

                {/* playback and submit controls */}
                {audioBlob && !isRecording && (
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '1rem',
                        marginBottom: '1rem'
                    }}>
                        <button
                            onClick={clearRecording}
                            style={{
                                width: '50px',
                                height: '50px',
                                background: 'linear-gradient(45deg, #9E9E9E, #757575)',
                                color: 'white',
                                borderRadius: '50%',
                                border: 'none',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                cursor: 'pointer',
                                fontSize: '0.7rem',
                                transition: 'all 0.3s ease',
                                fontWeight: 'bold'
                            }}
                        >
                            DELETE
                        </button>

                        <button
                            onClick={handleSubmit}
                            style={{
                                padding: '1rem 2rem',
                                background: 'linear-gradient(45deg, #4CAF50, #45a049)',
                                color: 'white',
                                borderRadius: '25px',
                                border: 'none',
                                fontWeight: '600',
                                cursor: 'pointer',
                                fontSize: '1rem',
                                transition: 'all 0.3s ease',
                                boxShadow: '0 8px 20px rgba(76, 175, 80, 0.3)',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.5rem'
                            }}
                        >
                            Send Recording
                        </button>
                    </div>
                )}

                {/* instructions */}
                <div style={{
                    color: 'rgba(255, 255, 255, 0.8)',
                    fontSize: '0.9rem',
                    lineHeight: '1.5'
                }}>
                    {!isRecording && !audioBlob && (
                        <p style={{ margin: 0 }}>
                            Click the microphone to start recording your message
                        </p>
                    )}
                    {isRecording && (
                        <p style={{ margin: 0 }}>
                            Recording in progress... Click STOP to finish or PAUSE to take a break
                        </p>
                    )}
                    {audioBlob && !isRecording && (
                        <div>
                            <p style={{ margin: '0 0 0.5rem 0' }}>
                                Recording complete!
                            </p>
                            <p style={{ margin: 0, fontSize: '0.8rem', opacity: 0.8 }}>
                                Review and click "Send Recording" to submit, or DELETE to clear
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </>
    );
};