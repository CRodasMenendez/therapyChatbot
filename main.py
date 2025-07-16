from transformers import AutoTokenizer, AutoModelForSequenceClassification
from stt import transcribe
import torch
import torch.nn.functional as F

# Load tokenizer and model from local directory
model_path = "./my-finetuned-model" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define the emotion labels model was trained on 
emotions = [
    'afraid', 'angry', 'annoyed', 'anticipating', 'anxious', 'apprehensive', 
    'ashamed', 'caring', 'confident', 'content', 'devastated', 'disappointed', 
    'disgusted', 'embarrassed', 'excited', 'faithful', 'furious', 'grateful', 
    'guilty', 'hopeful', 'impressed', 'jealous', 'joyful', 'lonely', 'nostalgic', 
    'prepared', 'proud', 'sad', 'sentimental', 'surprised', 'terrified', 'trusting'
]

def analyze_spoken_text():
    print("Starting emotional analysis of speech...")
    
    # Get transcribed text from the stt module 
    transcribed_text = transcribe()
    
    # Check if transcription was successful
    if transcribed_text and transcribed_text.strip():
        print(f"Input: '{transcribed_text}'")
        
        # Tokenize the text for the model (same way you did in training)
        inputs = tokenizer(transcribed_text, return_tensors='pt', truncation=True,padding=True, max_length=512)
        
        # Run the text through the model
        with torch.no_grad():  # Disable gradient calculation for inference (I don't need weights since im no longer training)
            outputs = model(**inputs)
        
        # Extract the raw logits (model's raw predictions)
        logits = outputs.logits
        
        # Convert logits to probabilities 
        probabilities = F.softmax(logits, dim=-1)
        
        # Get the predicted emotion (highest probability)
        predicted_emotion_id = torch.argmax(probabilities, dim=-1).item()
        predicted_emotion = emotions[predicted_emotion_id]
        
        # Get the confidence score for the predicted emotion
        confidence_score = probabilities[0][predicted_emotion_id].item()
        
        #print highest confidence emotion and how confident
        print(f"Predicted Emotion: {predicted_emotion.upper()}")
        print(f"Confidence: {confidence_score:.4f} ({confidence_score*100:.2f}%)")
        
        # Show top 5 most likely emotions
        top3_indices = torch.topk(probabilities, 3).indices[0]
        top3_probs = torch.topk(probabilities, 3).values[0]
        
        print(f"\n Top 3 Emotions")
        for rank in range(3):
        # Get the emotion ID and convert tensor to regular Python int
            emotion_id = top3_indices[rank].item()
    
            # Get the probability and convert tensor to regular Python float  
            probability = top3_probs[rank].item()
    
            # Look up the actual emotion name using the ID
            emotion_name = emotions[emotion_id]
    
            # Print with formatting
            rank_number = rank + 1  # Start counting from 1, not 0
            print(f"{rank_number}. {emotion_name:12} {probability:.3f} ({probability*100:.1f}%)")
        
        # Categorize emotions for easier understanding
        print(f"\n EMOTION CATEGORY ")
        emotion_categories = {
            'positive': ['caring', 'confident', 'content', 'excited', 'faithful', 'grateful', 
                        'hopeful', 'impressed', 'joyful', 'nostalgic', 'prepared', 'proud', 
                        'sentimental', 'surprised', 'trusting'],
            'negative': ['afraid', 'angry', 'annoyed', 'anxious', 'apprehensive', 'ashamed', 
                        'devastated', 'disappointed', 'disgusted', 'embarrassed', 'furious', 
                        'guilty', 'jealous', 'lonely', 'sad', 'terrified'],
            'neutral': ['anticipating']
        }
        
        category = 'unknown'
        for cat, emotion_list in emotion_categories.items():
            if predicted_emotion in emotion_list:
                category = cat
                break
        
        print(f"Emoption Category: {category.upper()}")
        
        return {
            'transcribed_text': transcribed_text,
            'predicted_emotion': predicted_emotion,
            'predicted_emotion_id': predicted_emotion_id,
            'confidence': confidence_score,
            'category': category,
            'top3_emotions': [(emotions[idx.item()], prob.item()) for idx, prob in zip(top3_indices, top3_probs)],
            'probabilities': probabilities,
            'logits': logits
        }
    else:
        print("No text was transcribed from the audio or transcription failed")
        return None

# Run the analysis
if __name__ == "__main__":
    result = analyze_spoken_text()