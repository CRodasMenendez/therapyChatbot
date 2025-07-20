from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from stt import transcribe
import torch
import torch.nn.functional as F
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#load tokenizer and model from local directory
emotion_tokenizer = AutoTokenizer.from_pretrained("./my-finetuned-model")
emotion_model = AutoModelForSequenceClassification.from_pretrained("./my-finetuned-model")

response_tokenizer = GPT2Tokenizer.from_pretrained("./my-finetuned-language-model")
response_model = GPT2LMHeadModel.from_pretrained("./my-finetuned-language-model")
response_tokenizer.pad_token = response_tokenizer.eos_token









# ---------- Function to simply analyze the sentiment of spoken text, made this function prior to starting the language model -----
def analyze_spoken_text():
    
    #get transcribed text from the stt module 
    transcribed_text = transcribe()
    
    #check if transcription was successful
    if transcribed_text and transcribed_text.strip():
        print(f"Input: '{transcribed_text}'")
        
        #tokenize the text for the model
        inputs = emotion_tokenizer(transcribed_text, return_tensors='pt', truncation=True,padding=True, max_length=512)
        
        #run the text through the model
        with torch.no_grad():  #disable gradient calculation for inference (I don't need weights since im no longer training)
            outputs = emotion_model(**inputs)
        
        #extract the raw logits (model's raw predictions)
        logits = outputs.logits
        
        #convert logits to probabilities 
        probabilities = F.softmax(logits, dim=-1)
        
        #get the predicted emotion (highest probability)
        predicted_emotion_id = torch.argmax(probabilities, dim=-1).item()
        predicted_emotion = emotions[predicted_emotion_id]
        
        #get the confidence score for the predicted emotion
        confidence_score = probabilities[0][predicted_emotion_id].item()
        
        #print highest confidence emotion and how confident
        print(f"Predicted Emotion: {predicted_emotion.upper()}")
        print(f"Confidence: {confidence_score:.4f} ({confidence_score*100:.2f}%)")
        
        #show top 5 most likely emotions
        top3_indices = torch.topk(probabilities, 3).indices[0]
        top3_probs = torch.topk(probabilities, 3).values[0]
        
        print(f"\n Top 3 Emotions")
        for rank in range(3):
        #get the emotion ID and convert tensor to regular Python int
            emotion_id = top3_indices[rank].item()
    
            #get the probability and convert tensor to regular Python float  
            probability = top3_probs[rank].item()
    
            #look up the actual emotion name using the ID
            emotion_name = emotions[emotion_id]
    
            #print with formatting
            rank_number = rank + 1  #start counting from 1, not 0
            print(f"{rank_number}. {emotion_name:12} {probability:.3f} ({probability*100:.1f}%)")
        
        #categorize emotions for easier understanding
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
    
    
    
# --------------------------- Functions to actually run the therapy session ----------------------------    
    
    
#function to analyze emotion using finetuned RoBERTa Model
def analyze_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    #get top emotion prediction
    emotion_id = torch.argmax(predictions, dim=-1).item()
    confidence = torch.max(predictions).item()
    emotion = emotions[emotion_id]
        
    return emotion, confidence


#function to generate response from language model
def generate_response(user_input, detected_emotion=None, confidence=0.0):
    #create prompt based off of emotion
    if detected_emotion and confidence > 0.6:
        #add emotion context to the prompt
        emotion_context = f"Emotion: {detected_emotion}\n"
        therapeutic_starter = response_strategies.get(detected_emotion, "")
            
        if therapeutic_starter:
            prompt = f"{emotion_context}User: {user_input}\nTherapist: {therapeutic_starter}"
        else:
            prompt = f"{emotion_context}User: {user_input}\nTherapist:"
    else:
        #standard prompt without emotion context
        prompt = f"User: {user_input}\nTherapist:"
        
    #tokenize and generate
    inputs = response_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(response_model.device) for k, v in inputs.items()}
        
    with torch.no_grad():
        outputs = response_model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 120,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=response_tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
        
    # Decode response
    generated_text = response_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
        
    # Clean up response
    if response:
        for ending in ['. ', '! ', '? ']:
            if ending in response:
                end_idx = response.find(ending) + 1
                response = response[:end_idx]
                break
        
    return response if response else "I'm here to listen. Can you tell me more about what you're experiencing?"
    
    
    
def therapy():
    #get transcribed text from the stt module 
    transcribed_text = transcribe()
    
    #check if transcription was successful
    if transcribed_text and transcribed_text.strip():
        print(f"Input: '{transcribed_text}'")
        
        #analyze sentiment of input
        emotion,confidence = analyze_emotion(transcribed_text)
        response = generate_response(transcribed_text, emotion, confidence)
        
        return {
            'detected_emotion': emotion,
            'confidence': confidence,
            'response': response,
            'used_emotion_context': confidence > 0.6
        }
        
        
#define the emotion labels model was trained on 
emotions = [
    'afraid', 'angry', 'annoyed', 'anticipating', 'anxious', 'apprehensive', 
    'ashamed', 'caring', 'confident', 'content', 'devastated', 'disappointed', 
    'disgusted', 'embarrassed', 'excited', 'faithful', 'furious', 'grateful', 
    'guilty', 'hopeful', 'impressed', 'jealous', 'joyful', 'lonely', 'nostalgic', 
    'prepared', 'proud', 'sad', 'sentimental', 'surprised', 'terrified', 'trusting'
]

#define response strategies for response model
response_strategies = {
    'angry': "I can hear the frustration and anger in your words. That sounds really difficult to deal with. When you feel this anger, what thoughts go through your mind?",
    'furious': "You sound extremely upset about this situation. Those feelings are completely valid. Can you help me understand what's making you feel most furious right now?",
    'anxious': "I can sense the anxiety you're experiencing. Workplace anxiety can be overwhelming, especially when it involves difficult relationships with supervisors. What aspect of work makes you feel most anxious?",
    'sad': "I hear the sadness in what you're sharing. It takes courage to express these feelings. What's been weighing most heavily on your heart lately?",
    'afraid': "Fear can be really difficult to sit with. You're showing strength by talking about this. What feels most frightening to you about this situation?",
    'lonely': "Feeling lonely can be incredibly painful. You're not alone in this conversation, and your feelings are completely valid. When do you notice the loneliness most?",
    'guilty': "Guilt can be such a heavy burden to carry. It sounds like you're being hard on yourself. What's making you feel most guilty right now?",
    'ashamed': "Shame can make us want to hide, but you're showing courage by sharing this. These feelings don't define your worth as a person. What's bringing up these feelings of shame?",
    'hopeful': "I can hear some hope in your voice, which is wonderful. Hope can be a powerful force for healing. What's giving you the most hope right now?",
    'grateful': "It's beautiful that you can find gratitude even during difficult times. That shows real resilience. What are you feeling most grateful for?",
    'excited': "I can feel your excitement! It's wonderful when we have positive things to look forward to. What's got you feeling so excited?",
    'content': "It sounds like you're in a good place right now. I'm glad to hear that. What's helping you feel content?",
    'confident': "I can hear the confidence in your voice. That's great to see. What's helping you feel so sure of yourself?",
    'disappointed': "Disappointment can be really hard to process. It sounds like something didn't go the way you hoped. What's been most disappointing for you?",
    'devastated': "That sounds absolutely overwhelming. I can only imagine how difficult this must be for you. Do you want to tell me more about what's making you feel this way?"
}        
        
    
        
        
    
    

#run the analysis
if __name__ == "__main__":
    while True:
        result = therapy()
        print(f"Detected Emotion: {result['detected_emotion']} (confidence: {result['confidence']:.2f})")
        print(f"Used Emotion Context: {result['used_emotion_context']}")
        print(f"Therapist: {result['response']}")
        
        