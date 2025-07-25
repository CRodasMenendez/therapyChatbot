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

    // start the recording timer
    const startTimer = () => {
        timerRef.current = setInterval(() => {
            setRecordingTime(prev => prev + 1);
        }, 1000);
    };

    // stop the recording timer
    const stopTimer = () => {
        if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
        }
    };

    // reset the recording timer
    const resetTimer = () => {
        stopTimer();
        setRecordingTime(0);
    };

    // start recording audio
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

    // stop recording audio
    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            setIsPaused(false);
            stopTimer();
        }
    };

    // pause recording audio
    const pauseRecording = () => {
        if (mediaRecorderRef.current && isRecording && !isPaused) {
            mediaRecorderRef.current.pause();
            setIsPaused(true);
            stopTimer();
        }
    };

    // resume recording audio
    const resumeRecording = () => {
        if (mediaRecorderRef.current && isRecording && isPaused) {
            mediaRecorderRef.current.resume();
            setIsPaused(false);
            startTimer();
        }
    };

    // clear current recording
    const clearRecording = () => {
        setAudioBlob(null);
        setError(null);
        chunksRef.current = [];
        resetTimer();

        if (isRecording) {
            stopRecording();
        }
    };

    // submit the recorded audio
    const handleSubmit = () => {
        if (audioBlob) {
            onAudioSubmit(audioBlob);
            clearRecording();
        }
    };

    // format recording time for display
    const formatRecordingTime = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div style={{ textAlign: 'center' }}>
            {/* error display */}
            {error && (
                <div style={{
                    padding: '0.75rem',
                    backgroundColor: '#fee2e2',
                    border: '1px solid #fecaca',
                    borderRadius: '0.5rem',
                    marginBottom: '1rem'
                }}>
                    <p style={{ color: '#dc2626', fontSize: '0.875rem', margin: 0 }}>{error}</p>
                </div>
            )}

            {/* main recording controls */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '1rem',
                marginBottom: '1rem'
            }}>
                {/* record/stop button */}
                {!isRecording ? (
                    <button
                        onClick={startRecording}
                        disabled={disabled}
                        style={{
                            width: '4rem',
                            height: '4rem',
                            backgroundColor: disabled ? '#9ca3af' : '#4f46e5',
                            color: 'white',
                            borderRadius: '50%',
                            border: 'none',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: disabled ? 'not-allowed' : 'pointer',
                            fontSize: '1.5rem'
                        }}
                    >
                        🎤
                    </button>
                ) : (
                    <button
                        onClick={stopRecording}
                        style={{
                            width: '4rem',
                            height: '4rem',
                            backgroundColor: '#dc2626',
                            color: 'white',
                            borderRadius: '50%',
                            border: 'none',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            fontSize: '1.5rem'
                        }}
                    >
                        ⏹️
                    </button>
                )}

                {/* pause/resume button */}
                {isRecording && (
                    <button
                        onClick={isPaused ? resumeRecording : pauseRecording}
                        style={{
                            width: '3rem',
                            height: '3rem',
                            backgroundColor: '#06b6d4',
                            color: 'white',
                            borderRadius: '50%',
                            border: 'none',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            fontSize: '1rem'
                        }}
                    >
                        {isPaused ? '▶️' : '⏸️'}
                    </button>
                )}
            </div>

            {/* recording timer */}
            {isRecording && (
                <div style={{ marginBottom: '1rem' }}>
                    <div style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        backgroundColor: '#f8fafc',
                        padding: '0.5rem 1rem',
                        borderRadius: '0.5rem'
                    }}>
                        <div style={{
                            width: '0.5rem',
                            height: '0.5rem',
                            backgroundColor: '#dc2626',
                            borderRadius: '50%'
                        }}></div>
                        <span style={{ fontFamily: 'monospace', color: '#1e293b' }}>
                            {formatRecordingTime(recordingTime)}
                        </span>
                        {isPaused && (
                            <span style={{ color: '#d97706', fontSize: '0.875rem' }}>(Paused)</span>
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
                            width: '2.5rem',
                            height: '2.5rem',
                            backgroundColor: '#6b7280',
                            color: 'white',
                            borderRadius: '50%',
                            border: 'none',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            fontSize: '1rem'
                        }}
                    >
                        🗑️
                    </button>

                    <button
                        onClick={handleSubmit}
                        style={{
                            padding: '0.75rem 1.5rem',
                            backgroundColor: '#10b981',
                            color: 'white',
                            borderRadius: '0.5rem',
                            border: 'none',
                            fontWeight: '500',
                            cursor: 'pointer'
                        }}
                    >
                        📤 Send Recording
                    </button>
                </div>
            )}

            {/* instructions for user */}
            <div style={{ color: '#64748b', fontSize: '0.875rem' }}>
                {!isRecording && !audioBlob && (
                    <p>Click the microphone to start recording your message</p>
                )}
                {isRecording && (
                    <p>Click the square to stop recording, or pause to take a break</p>
                )}
                {audioBlob && !isRecording && (
                    <p>Review your recording and click "Send Recording" to submit</p>
                )}
            </div>
        </div>
    );
};