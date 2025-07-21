from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from stt import transcribe
import torch
import torch.nn.functional as F
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#load tokenizer and model from local directory
emotion_tokenizer = AutoTokenizer.from_pretrained("./my-finetuned-model")
emotion_model = AutoModelForSequenceClassification.from_pretrained("./my-finetuned-model")

response_tokenizer = GPT2Tokenizer.from_pretrained("./my-updated-finetuned-language-model")
response_model = GPT2LMHeadModel.from_pretrained("./my-updated-finetuned-language-model")
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

#conversation memory

conversation_history = []
recent_emotions = []
session_topics = []

def add_to_memory(input, emotion, response):
    conversation_history.append({
        'user' : input,
        'emotion' : emotion,
        'response' : response
    })
    
    #only keep the last 5 emotions for context
    recent_emotions.append(emotion)
    if len(recent_emotions) > 5:
        recent_emotions.pop(0)
    
    #keyword detection for topics
    keywords = ['work', 'boss', 'job', 'family', 'relationship', 'anxiety', 'depression', 'stress']
    for keyword in keywords:
        if keyword.lower() in input.lower():
            if keyword not in session_topics:
                session_topics.append(keyword)
                

def get_conversation_context():
    if not conversation_history:
        return ""
    
    #get recent topics and emotions
    topics_str = ", ".join(session_topics[-3:]) if session_topics else "general"
    emotions_str = ", ".join(recent_emotions[-2:]) if len(recent_emotions) > 1 else ""
    
    context = f"Previous topics: {topics_str}. "
    if emotions_str:
        context += f"Recent emotions: {emotions_str}. "
    
    return context
    
        
    
    
    
#function to analyze emotion using finetuned RoBERTa Model
def analyze_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    #get top emotion prediction
    top3_indices = torch.topk(predictions, 3).indices[0]
    top3_probs = torch.topk(predictions, 3).values[0]
    
    conversational_responses = ['content', 'confident', 'prepared', 'trusting'] #filter out emotions that are just conversational
    
    emotion_id = top3_indices[0].item()
    confidence = top3_probs[0].item()
    emotion = emotions[emotion_id]
    
    #check second prediction if analyzed sentiment is just conversation with high confidence
    if emotion in conversational_responses and confidence > 0.8 and len(text.split()) > 15 and len(top3_indices) > 1:
        second_emotion = emotions[top3_indices[1].item()]
        second_confidence = top3_probs[1].item()
        
        if second_confidence > 0.3:
            emotion = second_emotion
            confidence = second_confidence
        
    return emotion, confidence


#function to generate response from language model
def generate_response(user_input, detected_emotion=None, confidence=0.0):
    #create response based off of emotion
    
    context = get_conversation_context()
    
    
    if detected_emotion and confidence > 0.6:
        therapeutic_starter = response_strategies.get(detected_emotion, "")
        
        if therapeutic_starter:
            prompt = f"Context: {context}User: {user_input}\nTherapist: {therapeutic_starter}"
        else:
            prompt = f"Context: {context}Emotion: {detected_emotion}\nUser: {user_input}\nTherapist:"
    else:
        prompt = f"Context: {context}User: {user_input}\nTherapist: I hear what you're saying."
    
        
    #tokenize and generate
    inputs = response_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(response_model.device) for k, v in inputs.items()}
        
    with torch.no_grad():
        outputs = response_model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 120,
            temperature=0.6,
            top_p=0.85,
            do_sample=True,
            pad_token_id=response_tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            repetition_penalty = 1.1
        )
        
    #decode response
    generated_text = response_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
        
    #clean up response
    if response:
        #remove context that may have leaked through
        if response.startswith("Context:"):
            response = response.split("Therapist:")[-1].strip()
            
        #find natural ending
        sentences = []
        current_sentence = ""
        
        for char in response:
            current_sentence += char
            if char in '.!?':
                sentences.append(current_sentence.strip())
                current_sentence = ""
                
        if sentences:
            if len(sentences) == 1:
                response = sentences[0]
            else:
                #if first sentence is very short, include second
                if len(sentences[0].split()) < 8 and len(sentences) > 1:
                    response = sentences[0] + " " + sentences[1]
                else:
                    response = sentences[0]
        
        #fallback cleaning
        if not response.endswith(('.', '!', '?')):
            last_punct = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
            if last_punct > 0:
                response = response[:last_punct + 1]
        
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
        
        add_to_memory(transcribed_text,emotion, response)
        
        return {
            'detected_emotion': emotion,
            'confidence': confidence,
            'response': response,
            'used_emotion_context': confidence > 0.6
        }
    else:
        print("No input detected, please try again \n")
        return None
        
        
    
        
        
    
    



#run the analysis
if __name__ == "__main__":
    while True:
        result = therapy()
        print(f"Detected Emotion: {result['detected_emotion']} (confidence: {result['confidence']:.2f})")
        print(f"Used Emotion Context: {result['used_emotion_context']}")
        print(f"Therapist: {result['response']}")
        
        