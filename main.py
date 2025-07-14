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

#install_NLTK_resources()













