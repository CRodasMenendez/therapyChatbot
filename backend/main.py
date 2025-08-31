# backend/main.py 
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import base64
import os
from typing import Dict, List, Optional
from datetime import datetime

# import AI components with error handling for deployment
try:
    from emotion_analyzer import EmotionAnalyzer
    from response_generator import ResponseGenerator
    from audio_processor import AudioProcessor
    print("loaded real AI components")
except ImportError as e:
    print(f"could not load AI components: {e}")
    print("using mock components for deployment")
    # you can add mock imports here if needed
    raise

app = FastAPI(title="AI Therapist API", version="1.0.0")

# get environment variables for production deployment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
PORT = int(os.getenv("PORT", 8080))

print(f"starting in {ENVIRONMENT} mode on port {PORT}")

# configure CORS for both development and production
if ENVIRONMENT == "production":
    # for gcp deployment
    allowed_origins = [
        "https://therapy-chatbot-wip.vercel.app",  # Vercel URL
        "https://*.vercel.app",  # allow vercel preview deployments
        "https://therapychatbot.xyz", #custom domain
        "https://www.therapychatbot.xyz" #www version of custom domain
    ]
else:
    # development origins
    allowed_origins = ["http://localhost:3000"]

print(f"allowed CORS origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize AI components with error handling
try:
    print("initializing AI components...")
    emotion_analyzer = EmotionAnalyzer()
    print("emotion analyzer loaded")
    
    response_generator = ResponseGenerator()
    print("response generator loaded")
    
    audio_processor = AudioProcessor()
    print("audio processor loaded")
    
    print("all AI components ready!")
except Exception as e:
    print(f"error initializing AI components: {e}")
    raise

# manage WebSocket connections and session data
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.sessions: Dict[str, dict] = {}
        print("connection manager initialized")

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
        print(f"new session connected: {session_id}")

    def disconnect(self, websocket: WebSocket, session_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if session_id in self.sessions:
            del self.sessions[session_id]
        print(f"session disconnected: {session_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "AI Therapist API is running on GCP",
        "environment": ENVIRONMENT,
        "port": PORT,
        "active_sessions": len(manager.sessions)
    }

@app.get("/")
async def root():
    return {
        "message": "AI Therapist API - deployed on GCP", 
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/api/process-audio")
async def process_audio(audio_file: UploadFile = File(...)):
    """process uploaded audio file and return transcription with emotion analysis"""
    try:
        print(f"processing audio file: {audio_file.filename}")
        audio_data = await audio_file.read()
        print(f"audio data size: {len(audio_data)} bytes")
        
        # transcribe audio using whisper
        transcription = audio_processor.transcribe_audio(audio_data)
        print(f"transcription: {transcription[:100]}...")
        
        if not transcription or not transcription.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        # analyze emotion using finetuned model
        emotion_result = emotion_analyzer.analyze_emotion(transcription)
        print(f"detected emotion: {emotion_result['emotion']} ({emotion_result['confidence']:.2f})")
        
        return {
            "transcription": transcription,
            "emotion": emotion_result["emotion"],
            "confidence": emotion_result["confidence"],
            "top_emotions": emotion_result["top_emotions"],
            "category": emotion_result["category"]
        }
        
    except Exception as e:
        print(f"error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/api/generate-response")
async def generate_response(request: dict):
    """generate therapeutic response based on user input and detected emotion"""
    try:
        user_input = request.get("user_input")
        detected_emotion = request.get("emotion")
        confidence = request.get("confidence", 0.0)
        session_id = request.get("session_id")
        
        print(f"generating response for emotion: {detected_emotion}")
        
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
        
        print(f"generated response: {response[:100]}...")
        
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
        print(f"error generating response: {e}")
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
                    print("processing websocket audio data")
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
                    print(f"websocket audio error: {e}")
                    error_response = {
                        "type": "error",
                        "content": f"Error processing audio: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(error_response))
            
            elif message_data["type"] == "text_message":
                print("processing websocket text message")
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
        print(f"websocket client {session_id} disconnected")

# startup event
@app.on_event("startup")
async def startup_event():
    print("AI Therapist API starting up on GCP...")
    print(f"environment: {ENVIRONMENT}")
    print(f"port: {PORT}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=ENVIRONMENT == "development",
        log_level="info"
    )