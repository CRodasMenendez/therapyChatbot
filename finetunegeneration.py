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

#load dataset
ds_mental_chat = load_dataset("ShenLab/MentalChat16K")

print(f"MentalChat16K: {len(ds_mental_chat['train'])} conversations")

#------------------- process MentalChat16K dataset -----------------------------
mental_chat_conversations = []
max_mental_chat = 10000  #limit to 10000 for faster training

for example in tqdm(ds_mental_chat['train'], desc="Processing MentalChat16K"):
    if len(mental_chat_conversations) >= max_mental_chat:
        break
    
    #extract user and therapist content
    user_input = example.get('input', '').strip()
    response = example.get('output', '').strip()
    
    #skip if no user input or response
    if not user_input or not response:
        continue
    
    #build conversation text
    conversation_text = f"User: {user_input}\nTherapist: {response}\n<|endoftext|>"
    
    #only add substantial conversations (more than 20 words)
    if len(conversation_text.split()) > 20:
        mental_chat_conversations.append(conversation_text)

print(f"Processed {len(mental_chat_conversations)} MentalChat16K conversations")

#split data into training and validation sets
train_texts, val_texts = train_test_split(
    mental_chat_conversations, 
    test_size=0.1,  # 10% for validation
    random_state=42
)

print(f"Training conversations: {len(train_texts)}")
print(f"Validation conversations: {len(val_texts)}")

#initialize GPT-2 model and tokenizer
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

#convert text to tokens
train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

#create pytorch dataset
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
    output_dir="./my-updated-finetuned-language-model", #where to save the model
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

#error handling for overnight training
try:
    #train the model
    trainer.train()

    #save the final model and tokenizer
    trainer.save_model("./my-updated-finetuned-language-model")
    tokenizer.save_pretrained("./my-updated-finetuned-language-model")
    print("Training completed successfully!")
    
except Exception as e:
    print(f"Training error: {e}")
    print("Saving current progress...")
    try:
        trainer.save_model("./my-updated-finetuned-language-model-backup")
        tokenizer.save_pretrained("./my-updated-finetuned-language-model-backup")
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