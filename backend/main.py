# backend/main.py
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import base64
from typing import Dict, List, Optional
from datetime import datetime

from emotion_analyzer import EmotionAnalyzer
from response_generator import ResponseGenerator
from audio_processor import AudioProcessor

app = FastAPI(title="AI Therapist API", version="1.0.0")

# allow React frontend to connect to this backend API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize AI components
emotion_analyzer = EmotionAnalyzer()
response_generator = ResponseGenerator()
audio_processor = AudioProcessor()

# manage WebSocket connections and session data
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.sessions: Dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        # store session data for conversation context
        self.sessions[session_id] = {
            "conversation_history": [],
            "recent_emotions": [],
            "session_topics": [],
            "started_at": datetime.now().isoformat()
        }

    def disconnect(self, websocket: WebSocket, session_id: str):
        self.active_connections.remove(websocket)
        if session_id in self.sessions:
            del self.sessions[session_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "AI Therapist API is running"}

@app.post("/api/process-audio")
async def process_audio(audio_file: UploadFile = File(...)):
    """process uploaded audio file and return transcription with emotion analysis"""
    try:
        audio_data = await audio_file.read()
        
        # transcribe audio using whisper
        transcription = audio_processor.transcribe_audio(audio_data)
        
        if not transcription or not transcription.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        # analyze emotion using finetuned model
        emotion_result = emotion_analyzer.analyze_emotion(transcription)
        
        return {
            "transcription": transcription,
            "emotion": emotion_result["emotion"],
            "confidence": emotion_result["confidence"],
            "top_emotions": emotion_result["top_emotions"],
            "category": emotion_result["category"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/api/generate-response")
async def generate_response(request: dict):
    """generate therapeutic response based on user input and detected emotion"""
    try:
        user_input = request.get("user_input")
        detected_emotion = request.get("emotion")
        confidence = request.get("confidence", 0.0)
        session_id = request.get("session_id")
        
        if not user_input:
            raise HTTPException(status_code=400, detail="User input is required")
        
        # get conversation context from session
        session_data = manager.sessions.get(session_id, {})
        
        # generate response using finetuned GPT-2 model
        response = response_generator.generate_response(
            user_input=user_input,
            detected_emotion=detected_emotion,
            confidence=confidence,
            conversation_history=session_data.get("conversation_history", []),
            recent_emotions=session_data.get("recent_emotions", [])
        )
        
        # update session conversation history
        if session_id and session_id in manager.sessions:
            manager.sessions[session_id]["conversation_history"].append({
                "user": user_input,
                "emotion": detected_emotion,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # keep track of recent emotions for context
            manager.sessions[session_id]["recent_emotions"].append(detected_emotion)
            if len(manager.sessions[session_id]["recent_emotions"]) > 5:
                manager.sessions[session_id]["recent_emotions"].pop(0)
        
        return {
            "response": response,
            "session_updated": session_id is not None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """retrieve session conversation history"""
    if session_id not in manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return manager.sessions[session_id]

@app.websocket("/ws/therapy/{session_id}")
async def websocket_therapy_session(websocket: WebSocket, session_id: str):
    """real-time therapy conversation over websocket"""
    await manager.connect(websocket, session_id)
    
    try:
        # send initial greeting
        welcome_message = {
            "type": "therapist_message",
            "content": "Hello! I'm here to listen and support you. How are you feeling today?",
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send_text(json.dumps(welcome_message))
        
        while True:
            # receive message from frontend
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "audio_data":
                try:
                    # decode audio data from base64
                    audio_bytes = base64.b64decode(message_data["audio"])
                    
                    # transcribe using whisper
                    transcription = audio_processor.transcribe_audio(audio_bytes)
                    
                    if transcription and transcription.strip():
                        # analyze emotion using finetuned RoBERTa
                        emotion_result = emotion_analyzer.analyze_emotion(transcription)
                        
                        # send transcription to frontend
                        transcription_response = {
                            "type": "transcription",
                            "content": transcription,
                            "emotion": emotion_result["emotion"],
                            "confidence": emotion_result["confidence"],
                            "timestamp": datetime.now().isoformat()
                        }
                        await websocket.send_text(json.dumps(transcription_response))
                        
                        # generate therapeutic response using finetuned GPT-2
                        session_data = manager.sessions.get(session_id, {})
                        response = response_generator.generate_response(
                            user_input=transcription,
                            detected_emotion=emotion_result["emotion"],
                            confidence=emotion_result["confidence"],
                            conversation_history=session_data.get("conversation_history", []),
                            recent_emotions=session_data.get("recent_emotions", [])
                        )
                        
                        # send therapist response to frontend
                        therapist_response = {
                            "type": "therapist_message",
                            "content": response,
                            "emotion_detected": emotion_result["emotion"],
                            "confidence": emotion_result["confidence"],
                            "timestamp": datetime.now().isoformat()
                        }
                        await websocket.send_text(json.dumps(therapist_response))
                        
                        # update session history
                        manager.sessions[session_id]["conversation_history"].append({
                            "user": transcription,
                            "emotion": emotion_result["emotion"],
                            "response": response,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # track recent emotions
                        manager.sessions[session_id]["recent_emotions"].append(emotion_result["emotion"])
                        if len(manager.sessions[session_id]["recent_emotions"]) > 5:
                            manager.sessions[session_id]["recent_emotions"].pop(0)
                    
                except Exception as e:
                    error_response = {
                        "type": "error",
                        "content": f"Error processing audio: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(error_response))
            
            elif message_data["type"] == "text_message":
                # handle typed messages
                user_input = message_data["content"]
                
                # analyze emotion from text
                emotion_result = emotion_analyzer.analyze_emotion(user_input)
                
                # generate response
                session_data = manager.sessions.get(session_id, {})
                response = response_generator.generate_response(
                    user_input=user_input,
                    detected_emotion=emotion_result["emotion"],
                    confidence=emotion_result["confidence"],
                    conversation_history=session_data.get("conversation_history", []),
                    recent_emotions=session_data.get("recent_emotions", [])
                )
                
                # send response
                therapist_response = {
                    "type": "therapist_message",
                    "content": response,
                    "emotion_detected": emotion_result["emotion"],
                    "confidence": emotion_result["confidence"],
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(therapist_response))
                
                # update session
                manager.sessions[session_id]["conversation_history"].append({
                    "user": user_input,
                    "emotion": emotion_result["emotion"],
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
        print(f"Client {session_id} disconnected")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )