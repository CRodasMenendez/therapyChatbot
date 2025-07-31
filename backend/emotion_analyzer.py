# backend/emotion_analyzer.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os

class EmotionAnalyzer:
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        model_path = "CRodas/therapist-emotion-model"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        #emotion labels the model was trained on
        self.emotions = [
            'afraid', 'angry', 'annoyed', 'anticipating', 'anxious', 'apprehensive', 
            'ashamed', 'caring', 'confident', 'content', 'devastated', 'disappointed', 
            'disgusted', 'embarrassed', 'excited', 'faithful', 'furious', 'grateful', 
            'guilty', 'hopeful', 'impressed', 'jealous', 'joyful', 'lonely', 'nostalgic', 
            'prepared', 'proud', 'sad', 'sentimental', 'surprised', 'terrified', 'trusting'
        ]
        
        #categorize emotions for response context
        self.emotion_categories = {
            'positive': ['caring', 'confident', 'content', 'excited', 'faithful', 'grateful', 
                        'hopeful', 'impressed', 'joyful', 'nostalgic', 'prepared', 'proud', 
                        'sentimental', 'surprised', 'trusting'],
            'negative': ['afraid', 'angry', 'annoyed', 'anxious', 'apprehensive', 'ashamed', 
                        'devastated', 'disappointed', 'disgusted', 'embarrassed', 'furious', 
                        'guilty', 'jealous', 'lonely', 'sad', 'terrified'],
            'neutral': ['anticipating']
        }
        
        #emotions that might indicate general conversation rather than strong emotion
        self.conversational_responses = ['content', 'confident', 'prepared', 'trusting']
        
        print("emotion analyzer initialized successfully")
    
    def analyze_emotion(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                'emotion': 'unknown',
                'confidence': 0.0,
                'category': 'unknown',
                'top_emotions': [],
                'all_probabilities': []
            }
        
        # tokenize text for model input
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True,
            padding=True, 
            max_length=512
        )
        
        #run inference without gradient calculation
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        #get probabilities from logits
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        
        #get top 3 emotion predictions
        top3_values, top3_indices = torch.topk(probabilities, 3)
        top3_indices = top3_indices[0]
        top3_probs = top3_values[0]
        
        #primary emotion prediction
        emotion_id = top3_indices[0].item()
        confidence = top3_probs[0].item()
        emotion = self.emotions[emotion_id]
        
        #check if should use second prediction for better accuracy
        if (emotion in self.conversational_responses and 
            confidence > 0.8 and 
            len(text.split()) > 15 and 
            len(top3_indices) > 1):
            
            second_emotion = self.emotions[top3_indices[1].item()]
            second_confidence = top3_probs[1].item()
            
            if second_confidence > 0.3:
                emotion = second_emotion
                confidence = second_confidence
        
        # determine emotion category
        category = 'unknown'
        for cat, emotion_list in self.emotion_categories.items():
            if emotion in emotion_list:
                category = cat
                break
        
        #prepare top emotions for frontend
        top_emotions = []
        for i in range(min(3, len(top3_indices))):
            emotion_name = self.emotions[top3_indices[i].item()]
            probability = top3_probs[i].item()
            top_emotions.append({
                'emotion': emotion_name,
                'probability': probability,
                'percentage': probability * 100
            })
        
        #prepare all probabilities for detailed analysis
        all_probabilities = []
        for i, prob in enumerate(probabilities[0]):
            all_probabilities.append({
                'emotion': self.emotions[i],
                'probability': prob.item(),
                'percentage': prob.item() * 100
            })
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'category': category,
            'top_emotions': top_emotions,
            'all_probabilities': all_probabilities,
            'raw_logits': logits.tolist(),
            'text_length': len(text.split())
        }