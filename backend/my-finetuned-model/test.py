import torch
import os
os.environ["WANDB_DISABLED"] = "true"
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
import json
import re
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

#set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#load datasets
ds_ed = load_dataset("Estwld/empathetic_dialogues_llm")
ds_counsel_chat = load_dataset("nbertagnolli/counsel-chat")

print(f"EmpatheticDialogues: {len(ds_ed['train'])} conversations")
print(f"Counsel Chat: {len(ds_counsel_chat['train'])} conversations")

#------------------- process empathetic dialogues dataset -----------------------------
empathetic_conversations = []
max_empathetic = 5000  #limit to 5000 for faster training

for example in tqdm(ds_ed['train'], desc="Processing Empathetic Dialogues"):
    if len(empathetic_conversations) >= max_empathetic:
        break
    
    #conversation metadata
    conv_id = example['conv_id']
    situation = example.get('situation', '')
    emotion = example.get('emotion', '')
    conversations = example.get('conversations', [])
    
    #skip if no conversations
    if not conversations:
        continue
    
    #build conversation text with context
    conversation_text = ""
    
    #add 'context'
    if situation:
        conversation_text += f"Context: {situation}\n"
    if emotion:
        conversation_text += f"Emotion: {emotion}\n"
    
    #process each turn in the conversation
    for turn in conversations:
        role = turn.get('role', 'unknown')
        content = turn.get('content', '').strip()
        
        if content:
            conversation_text += f"{role}: {content}\n"
    
    #add end-of-text token
    conversation_text += "<|endoftext|>"
    
    #only add substantial conversations (more than 10 words)
    if len(conversation_text.split()) > 10:
        empathetic_conversations.append(conversation_text)

print(f"Processed {len(empathetic_conversations)} Empathetic Dialogues conversations")

#process counsel chat dataset
counsel_conversations = []
max_counsel = 2000 

for i, example in enumerate(tqdm(ds_counsel_chat['train'], desc="Processing Counsel Chat")):
    if i >= max_counsel:
        break
    
    #extract question and answer fields 
    question_title = example.get('questionTitle', '')
    question_title = question_title.strip() if question_title else ''
    
    question_text = example.get('questionText', '')
    question_text = question_text.strip() if question_text else ''
    
    answer_text = example.get('answerText', '')
    answer_text = answer_text.strip() if answer_text else ''
    
    topic = example.get('topic', '')
    topic = topic.strip() if topic else ''
    
    #build conversation text
    conversation_text = ""
    
    #add topic context if available
    if topic:
        conversation_text += f"Topic: {topic}\n"
    
    #combine question title and text for user input
    user_input = ""
    if question_title and question_text:
        user_input = f"{question_title} - {question_text}"
    elif question_text:
        user_input = question_text
    elif question_title:
        user_input = question_title
    
    if user_input:
        conversation_text += f"User: {user_input}\n"
    
    #add therapist response
    if answer_text:
        conversation_text += f"Therapist: {answer_text}\n"
    
    #add end-of-text token
    conversation_text += "<|endoftext|>"
    
    #only add substantial conversations (more than 20 words for Q&A format)
    if len(conversation_text.split()) > 20:
        counsel_conversations.append(conversation_text)

print(f"Processed {len(counsel_conversations)} Counsel Chat conversations")

#combine all conversations for training
all_conversations = empathetic_conversations + counsel_conversations
print(f"\nTotal conversations for training: {len(all_conversations)}")

#split data into training and validation sets
train_texts, val_texts = train_test_split(
    all_conversations, 
    test_size=0.1,  # 10% for validation
    random_state=42
)

print(f"Training conversations: {len(train_texts)}")
print(f"Validation conversations: {len(val_texts)}")

#initialize GPT-2 model and tokenizer
print("\nInitializing model and tokenizer...")
model_name = "gpt2" #gpt-2 base model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

#set padding token
tokenizer.pad_token = tokenizer.eos_token

#tokenization function
def tokenize_function(examples):
    tokens = tokenizer(
        examples,
        truncation=True, 
        padding=True,  
        max_length=512, 
        return_tensors="pt"
    )

    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

print("Tokenizing datasets...")

#convert text to tokens
train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

#create pytorch datasets
class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

#create dataset objects
train_dataset = ConversationDataset(train_encodings)
val_dataset = ConversationDataset(val_encodings)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

#data collator for handling batches during training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  #false = causal language modeling (predict next word)
)

#training configuration 
training_args = TrainingArguments(
    output_dir="./my-finetuned-language-model", #where to save the model
    overwrite_output_dir=True,                  #overwrite existing output
    num_train_epochs=4,                         #4 epochs 
    per_device_train_batch_size=2,              #2 for stability  
    per_device_eval_batch_size=4,               #larger eval batch 
    gradient_accumulation_steps=2,              #simulate batch_size = 4
    warmup_steps=100,                           #learning rate warmup steps
    logging_steps=50,                           #how often to log training info
    save_steps=250,                             #save frequency
    eval_steps=250,                             #eval frequency
    eval_strategy="steps",                      #evaluate based on steps (not epochs)
    save_strategy="steps",                      #save based on steps
    load_best_model_at_end=True,                #load best model when training ends
    metric_for_best_model="eval_loss",          #use validation loss to pick best model
    greater_is_better=False,                    #lower loss is better
    report_to=None,                             #don't use wandb/tensorboard
    dataloader_pin_memory=False,                #helps with memory issues on macOS
    dataloader_num_workers=0,                   #to prevent macOS issues
)

#initialize the trainer
trainer = Trainer(
    model=model,                    #model
    args=training_args,             #training configuration
    train_dataset=train_dataset,    #training data
    eval_dataset=val_dataset,       #validation data
    data_collator=data_collator,    #how to batch the data
    tokenizer=tokenizer,            #tokenizer for text processing
)

print("\nStarting training...")
print("Estimated training time: 3-4 hours")

#error handling for overnight training
try:
    #train the model
    trainer.train()

    #save the final model and tokenizer
    trainer.save_model("./my-finetuned-language-model")
    tokenizer.save_pretrained("./my-finetuned-language-model")
    print("Training completed successfully!")
    
except Exception as e:
    print(f"Training error: {e}")
    print("Saving current progress...")
    try:
        trainer.save_model("./my-finetuned-language-model-backup")
        tokenizer.save_pretrained("./my-finetuned-language-model-backup")
        print("Backup saved!")
    except:
        print("Could not save backup")

#-------------------------------- testing finetuned model ----------------------------
print("\n ----------------------- Testing: ---------------------------------")

try:
    model.eval()  #set model to evaluation mode

    #test prompt
    test_prompt = "User: I've been feeling really anxious lately about work.\nTherapist:"

    #convert prompt to tokens
    inputs = tokenizer.encode(test_prompt, return_tensors="pt")

    #generate response
    with torch.no_grad():  #don't compute gradients for inference
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,    #generate up to 100 new tokens
            num_return_sequences=1,              #generate 1 response
            temperature=0.7,                     #controls randomness (0.0 = deterministic, 1.0 = very random)
            do_sample=True,                      #use sampling instead of greedy decoding
            pad_token_id=tokenizer.eos_token_id  #padding token
        )

    #convert tokens back to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Input: {test_prompt}")
    print(f"Generated response: {generated_text[len(test_prompt):]}")
    
except Exception as e:
    print(f"Testing error: {e}")
    print("Model may not have completed training")

print("Training session complete!")