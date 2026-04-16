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
from huggingface_hub import login

nltk.download('punkt_tab')


nltk.download('punkt')

#utils
from utils.utils import load_global_vocab, rule_based_annotate, optimized_annotate, create_huggingface_dataset, decoding_labels, tokenize_and_align_labels
from load_and_label import loading_label_data

label_to_id = {
    "O": 0,
    "B-ACTOR": 1, "I-ACTOR": 2,
    "B-SYSTEM": 3, "I-SYSTEM": 4,
    "B-PHASE": 5, "I-PHASE": 6,
    "B-TRIGGER": 7, "I-TRIGGER": 8,
    "B-OUTCOME": 9, "I-OUTCOME": 10
}
label_list = ["O","B-ACTOR", "I-ACTOR","B-SYSTEM", "I-SYSTEM","B-PHASE", "I-PHASE","B-TRIGGER", "I-TRIGGER","B-OUTCOME", "I-OUTCOME"]

login()


def train_model():
    #load labeled dataset
    dataset = loading_label_data()

    model_name = ""
    repo_name = ""

    #trigger if we want to create our own dataset or if we want to use the zipped one
    #label names

    label_name = decoding_labels(label_list=label_list)

    # Mappings for the model config
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )

    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    
    training_args = TrainingArguments(
        output_dir=repo_name,          
        eval_strategy="epoch",   
        eval_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=32, 
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        push_to_hub=True,               
        hub_model_id=repo_name,         #your repo name - you can add it here to push to hugging face
        report_to="none"            #you can add wandb
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # 5. Train and Push
    trainer.train()

    # Pushes the final model, tokenizer, and a draft model card to the Hub
    trainer.push_to_hub(commit_message="End of training")

    return model, tokenizer, dataset
    



    



