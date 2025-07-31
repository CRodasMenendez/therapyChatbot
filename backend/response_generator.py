# backend/response_generator.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List, Dict, Optional

class ResponseGenerator:
    def __init__(self):
        
        model_path = "CRodas/therapist-response-model"
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        
        #set pad token since GPT-2 doesn't have one by default
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        #therapeutic response strategies for different emotions
        self.response_strategies = {
            'angry': "I can hear the frustration and anger in your words. That sounds really difficult to deal with. When you feel this anger, what thoughts go through your mind?",
            'furious': "You sound extremely upset about this situation. Those feelings are completely valid. Can you help me understand what's making you feel most furious right now?",
            'anxious': "I can sense the anxiety you're experiencing. Anxiety can be overwhelming, especially when it involves difficult situations. What aspect makes you feel most anxious?",
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
            'devastated': "That sounds absolutely overwhelming. I can only imagine how difficult this must be for you. Do you want to tell me more about what's making you feel this way?",
            'terrified': "That level of fear sounds truly overwhelming. I'm here with you right now, and you're safe in this space. Can you tell me what's making you feel so terrified?",
            'joyful': "I can really feel the joy in what you're sharing! It's wonderful to hear you feeling this way. What's bringing you this happiness?",
            'nostalgic': "There's something beautiful about those memories you're connecting with. Nostalgia can bring both warmth and longing. What are you remembering?"
        }
        
        print("Response generator initialized successfully!")
    
    def generate_response(self, 
                        user_input: str, 
                        detected_emotion: Optional[str] = None, 
                        confidence: float = 0.0,
                        conversation_history: List[Dict] = None,
                        recent_emotions: List[str] = None) -> str:
        if not user_input or not user_input.strip():
            return "I'm here to listen. Can you tell me more about what you're experiencing?"
        
        #build context from conversation history and recent emotions
        context = self._build_context(conversation_history, recent_emotions)
        
        #create prompt based on emotion detection confidence
        if detected_emotion and confidence > 0.6:
            #high confidence emotion detection --> use emotion-specific strategy
            therapeutic_starter = self.response_strategies.get(detected_emotion, "")
            
            if therapeutic_starter:
                prompt = f"Context: {context}User: {user_input}\nTherapist: {therapeutic_starter}"
            else:
                prompt = f"Context: {context}Emotion: {detected_emotion}\nUser: {user_input}\nTherapist:"
        else:
            #low confidence or no emotion --> use general therapeutic approach
            prompt = f"Context: {context}User: {user_input}\nTherapist: I hear what you're saying."
        
        #generate response using the model
        response = self._generate_with_model(prompt)
        
        #clean and validate the response
        cleaned_response = self._clean_response(response, prompt)
        
        return cleaned_response
    
    def _build_context(self, 
                    conversation_history: List[Dict] = None, 
                    recent_emotions: List[str] = None) -> str:

        context_parts = []
        
        #add recent topics from conversation history
        if conversation_history:
            #extract keywords from recent conversations for topic continuity
            recent_conversations = conversation_history[-3:]
            topics = []
            
            #keyword extraction for common therapy topics
            keywords = ['work', 'boss', 'job', 'family', 'relationship', 'anxiety', 
                    'depression', 'stress', 'school', 'friends', 'money', 'health']
            
            for conv in recent_conversations:
                user_text = conv.get('user', '').lower()
                for keyword in keywords:
                    if keyword in user_text and keyword not in topics:
                        topics.append(keyword)
            
            if topics:
                context_parts.append(f"Previous topics: {', '.join(topics[-3:])}")
        
        #add recent emotions for emotional continuity
        if recent_emotions:
            unique_recent = []
            for emotion in reversed(recent_emotions):
                if emotion not in unique_recent:
                    unique_recent.append(emotion)
                if len(unique_recent) >= 2:
                    break
            
            if unique_recent:
                context_parts.append(f"Recent emotions: {', '.join(reversed(unique_recent))}")
        
        return ". ".join(context_parts) + ". " if context_parts else ""
    
    def _generate_with_model(self, prompt: str) -> str:
        try:
            #tokenize the prompt
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            
            #move inputs to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            #generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 120,
                    temperature=0.6,
                    top_p=0.85,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )
            
            #decode the generated response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            #extract only the new part after the prompt
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having trouble finding the right words right now. Can you tell me more about what you're feeling?"
    
    def _clean_response(self, response: str, original_prompt: str) -> str:
        if not response:
            return "I'm here to listen. Can you tell me more about what you're experiencing?"
        
        #remove any leaked context from the response
        if response.startswith("Context:"):
            response = response.split("Therapist:")[-1].strip()
        
        #handle multiple sentences
        sentences = []
        current_sentence = ""
        
        for char in response:
            current_sentence += char
            if char in '.!?':
                sentence = current_sentence.strip()
                if sentence:
                    sentences.append(sentence)
                current_sentence = ""
        
        #add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        if sentences:
            #strategic sentence selection
            if len(sentences) == 1:
                response = sentences[0]
            else:
                #if first sentence is very short include second for completeness
                if len(sentences[0].split()) < 8 and len(sentences) > 1:
                    response = sentences[0] + " " + sentences[1]
                else:
                    response = sentences[0]
        
        #ensure response ends with proper punctuation
        if response and not response.endswith(('.', '!', '?')):
            last_punct = max(
                response.rfind('.'), 
                response.rfind('!'), 
                response.rfind('?')
            )
            if last_punct > 0:
                response = response[:last_punct + 1]
            else:
                response = response.rstrip() + "."
        
        #final validation, ensure meaningful response
        if not response or len(response.split()) < 3:
            return "I'm here to listen. Can you tell me more about what you're experiencing?"
        
        return response