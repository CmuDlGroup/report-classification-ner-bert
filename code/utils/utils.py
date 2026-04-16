import json
import nltk
from nltk.tokenize import word_tokenize
from datasets import Dataset, DatasetDict, Sequence, ClassLabel
import pandas as pd
import torch
from evaluate import load
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader


def load_global_vocab(file_path):
    """Loads the aviation vocabulary from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Defaulting to empty vocab.")
        return {"ACTOR": [], "SYSTEM": [], "PHASE": [], "TRIGGER": [], "OUTCOME": []}
    except json.JSONDecodeError:
        print(f"Error: {file_path} contains invalid JSON.")
        return {}


def rule_based_annotate(row, GLOBAL_VOCAB, label_to_id):
    text = str(row['narrative'])
    tokens = word_tokenize(text)
    
    ner_tags = [0] * len(tokens)

    
    entities_to_check = {
        "ACTOR": [str(row['operator'])],
        "SYSTEM": [str(row['aircraft_type'])],
        "PHASE": [str(row['phase'])]
    }

    
    for label_type, words in GLOBAL_VOCAB.items():
        entities_to_check.setdefault(label_type, []).extend(words)

    for label_type, search_list in entities_to_check.items():
        
        sorted_search = sorted([str(s) for s in search_list if len(str(s)) > 2],
                               key=len, reverse=True)

        for entity_str in sorted_search:
            if not entity_str or entity_str.lower() == 'nan':
                continue

            entity_tokens = word_tokenize(entity_str)
            n_entity = len(entity_tokens)

            for i in range(len(tokens) - n_entity + 1):
                
                if ner_tags[i] == 0:
                    if [t.lower() for t in tokens[i:i+n_entity]] == [et.lower() for et in entity_tokens]:
                        ner_tags[i] = label_to_id[f"B-{label_type}"]
                        for j in range(1, n_entity):
                            ner_tags[i + j] = label_to_id[f"I-{label_type}"]

    return tokens, ner_tags

def optimized_annotate(text,label_to_id, lookup, phrases):
    if not isinstance(text, str) or not text.strip():
        return [], []
    
    # Tokenize and initialize with the ID for "O" (0)
    tokens = text.split() 
    ner_tags = [label_to_id["O"]] * len(tokens)
    
    text_lower = text.lower()
    tokens_lower = [t.lower().strip('.,!?;:()') for t in tokens]

    
    for pattern, category in phrases:
        for match in pattern.finditer(text_lower):
            start_char, end_char = match.span()
            
           
            start_token_idx = text_lower[:start_char].count(' ')
            num_words_in_match = match.group().count(' ') + 1
            
            for i in range(num_words_in_match):
                idx = start_token_idx + i
                if idx < len(ner_tags):
                    if i == 0:
                        ner_tags[idx] = label_to_id[f"B-{category}"]
                    else:
                        ner_tags[idx] = label_to_id[f"I-{category}"]

    #we avoid repeating what we already have
    for i, t_low in enumerate(tokens_lower):
        if ner_tags[i] == 0 and t_low in lookup:
            category = lookup[t_low]
            ner_tags[i] = label_to_id[f"B-{category}"]
                
    return tokens, ner_tags

def create_huggingface_dataset(df: pd.DataFrame):
    
    #Convert to Hugging Face Dataset format
    raw_ds = Dataset.from_pandas(df[['uid', 'tokens', 'ner_tags']])

    #Split into 80% Train, 10% Validation, 10% Test
    train_testvalid = raw_ds.train_test_split(test_size=0.2)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

    dataset = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })

    return dataset

def decoding_labels(label_list: list):
    

    
    custom_ner_features = Sequence(
        feature=ClassLabel(
            num_classes=11,
            names=label_list
        )
    )

    label_names = custom_ner_features.feature.names

    return label_names

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
        
                label_ids.append(label[word_idx])
            else:
                
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def evaluate_models_on_test(model_dict, raw_test_dataset):
    """
    model_dict: { "model_name": (model, tokenizer), ... }
    raw_test_dataset: The un-tokenized dataset (containing 'tokens' and 'ner_tags')
    """
    metric = load("seqeval")
    final_results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, (model, tokenizer) in model_dict.items():
        print(f"\n--- Preparing and Evaluating {name} ---")

        # 1. Tokenize and Align inside the function for this specific model
        tokenized_test = raw_test_dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer),
            batched=True,
            remove_columns=raw_test_dataset.column_names
        )

        # 2. Setup DataLoader
        data_collator = DataCollatorForTokenClassification(tokenizer)
        test_loader = DataLoader(
            tokenized_test,
            batch_size=16,
            collate_fn=data_collator
        )

        model.to(device)
        model.eval()

        all_predictions = []
        all_labels = []

        # 3. Batch Inference Loop
        for batch in tqdm(test_loader, desc=f"Inference"):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            # 4. Alignment & Filtering
            for i in range(len(labels)):
                # We KEEP "O" here so lengths match for seqeval
                # Seqeval will handle the "O" exclusion automatically in step 5
                true_predictions = [
                    model.config.id2label[p]
                    for (p, l) in zip(predictions[i], labels[i]) if l != -100
                ]
                true_labels = [
                    model.config.id2label[l]
                    for (p, l) in zip(predictions[i], labels[i]) if l != -100
                ]

                all_predictions.append(true_predictions)
                all_labels.append(true_labels)

        # 5. Compute Metrics
        # IMPORTANT: 'overall_f1' ignores 'O'. 'overall_accuracy' includes 'O'.
        results = metric.compute(predictions=all_predictions, references=all_labels)

        final_results[name] = {
            "Precision (Entities Only)": results["overall_precision"],
            "Recall (Entities Only)": results["overall_recall"],
            "F1 (Entities Only)": results["overall_f1"],
            "Accuracy (Global)": results["overall_accuracy"]
        }

    return pd.DataFrame(final_results).T, all_predictions, all_labels, metric

