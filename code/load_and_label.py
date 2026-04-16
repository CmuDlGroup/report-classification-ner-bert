import pandas as pd
import numpy as np
import torch
import pandas as pd
import re
import json
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)

import networkx as nx
import matplotlib.pyplot as plt
import zipfile
import nltk
from nltk.tokenize import word_tokenize
import os

nltk.download('punkt_tab')


nltk.download('punkt')

#utils
from utils.utils import load_global_vocab, rule_based_annotate, optimized_annotate, create_huggingface_dataset, decoding_labels, tokenize_and_align_labels


label_to_id = {
    "O": 0,
    "B-ACTOR": 1, "I-ACTOR": 2,
    "B-SYSTEM": 3, "I-SYSTEM": 4,
    "B-PHASE": 5, "I-PHASE": 6,
    "B-TRIGGER": 7, "I-TRIGGER": 8,
    "B-OUTCOME": 9, "I-OUTCOME": 10
}
label_list = ["O","B-ACTOR", "I-ACTOR","B-SYSTEM", "I-SYSTEM","B-PHASE", "I-PHASE","B-TRIGGER", "I-TRIGGER","B-OUTCOME", "I-OUTCOME"]



def loading_label_data():
    

    #trigger if we want to create our own dataset or if we want to use the zipped one
    useData = True
    savePath = "data"

    if useData:
        # Path to your zip file
        zip_path = "data/data_ner_dataset.zip"
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("data_ner_dataset")

        # Load the unzipped folder
        #this looks for the dataset_dict.json file automatically
        dataset = load_from_disk("data_ner_dataset")
        
    elif not useData:
        dataset = create_huggingface_dataset(df=df)

        df = pd.read_csv("aviation_data/asn_scraped_ds.csv")
    
        #Create the text column
        df["text"] = df["narrative"].fillna("") + " " + df["category"].fillna("")

        
        df = df[["text"]].dropna().reset_index(drop=True)

        
        # Load your vocab from the JSON
        with open("aviation_data/NER_labels_aviation.json", 'r') as f:
            GLOBAL_VOCAB = json.load(f)
        
        #build an optimized lookups table
        lookup = {}
        phrases = []

        for category, words in GLOBAL_VOCAB.items():
            for word in words:
                word_lower = word.lower().strip()
                if ' ' in word_lower:
                    
                    phrases.append((re.compile(rf'\b{re.escape(word_lower)}\b'), category))
                else:
                    lookup[word_lower] = category
        
        #create NER tags and token fields in teh Dataframe
        
        text_data = df['narrative'].astype(str).tolist()
        results = [optimized_annotate(t) for t in text_data]

        df['tokens'] = [r[0] for r in results]
        df['ner_tags'] = [r[1] for r in results]
        
        os.makedirs(os.path.dirname(savePath), exist_ok=True)

        #Save the DatasetDict directly to Disk in the root folder
        dataset.save_to_disk(savePath)
    
    return dataset



    



