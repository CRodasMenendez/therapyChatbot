import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('ggplot')

import nltk


#function to check if NLTK resources are installed and if not, it installs them
def install_NLTK_resources():
    resources = {
        'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'punkt_tab': 'tokenizers/punkt_tab',
        'maxent_ne_chunker_tab': 'chunkers/maxent_ne_chunker_tab',
        'words': 'corpora/words',
        'vader_lexicon': 'sentiment/vader_lexicon'
    }

    for name, path in resources.items():
        try:
            nltk.data.find(path)
            print(f"{name} already available.")
        except LookupError:
            print(f"Downloading {name}...")
            nltk.download(name)

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from transformers import pipeline


install_NLTK_resources()

from datasets import load_dataset
from torch.utils.data import DataLoader

# --------------------------- Process Empathetic Dialogues Dataset --------------------------
ds_ed = load_dataset("Estwld/empathetic_dialogues_llm")
#print(ds_ed)

#tokenizer object
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

#pre-process the emotions field
emotions = sorted(set(ds_ed['train']['emotion']))
emotion_to_id = {e: i for i,e in enumerate(emotions)} #gives each emotion a numeric ID for training and tokenization


#function to tokenize data
def roberta_tokenization(batch):
    # batch argument - Dictionary containing lists of data from the dataset 
    # batch['situation'] - List of situation text strings
    # batch['emotion'] - List of emotion labels (strings)
    
    # Returns Dictionary with tokenized data ready for model training:
    

    tokens = roberta_tokenizer(batch['situation'], padding='max_length',truncation=True, max_length = 512) 
    # batch['situation'] is a list like: ["I feel sad today", "I'm happy", ...]
    
    # Convert string emotion labels to numeric IDs 
    tokens['labels'] = [emotion_to_id[emotion] for emotion in batch['emotion']]
    
    return tokens



#tokenize dataset
tokenized_ed = ds_ed.map(roberta_tokenization, batched = True)
tokenized_ed.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels']) #format data
#print(tokenized_ed)
training_dataset_ed = tokenized_ed['train']
test_dataset_ed = tokenized_ed['test']
valid_dataset_ed = tokenized_ed['valid']


# print(ds_ed)
# print(ds_ed['train'][0])

train_dataloader_ed = DataLoader(training_dataset_ed, shuffle = True, batch_size = 8)
valid_dataloader_ed = DataLoader(valid_dataset_ed, batch_size = 8)

# -------------------------- Fine tune the roberta model -----------------------------

from transformers import Trainer, TrainingArguments
from torch.optim import AdamW


#load pretrained model 

analysis_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels = len(emotions))

training_arguments = TrainingArguments(
    output_dir = "./roberta_results",
    eval_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = 3,
    weight_decay = 0.01
)

trainer = Trainer(
    model = analysis_model,
    args = training_arguments,
    train_dataset = training_dataset_ed,
    eval_dataset = valid_dataset_ed,
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)

predictions = trainer.predict(valid_dataset_ed)
print(predictions)

analysis_model.save_pretrained('./my-finetuned-model')
roberta_tokenizer.save_pretrained('./my-finetuned-model')


# -------------------------------- downloading the fine-tuned model ---------------------
import shutil
import os

# Create a zip file of trained model
zip_filename = 'my-finetuned-model'
shutil.make_archive(zip_filename, 'zip', './my-finetuned-model')

print(f"Model saved and zipped as: {zip_filename}.zip")
print(f"Full path: {os.path.abspath(zip_filename + '.zip')}")