import numpy as np
import pandas as pd
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# MFT 01
class DataLoad:
    """
    Class to read in the BABE Dataset
    """

    @staticmethod
    def read_babe():
        df = pd.read_excel(
            "data/final_labels_SG2.xlsx",
        )
        lst = []
        for index, row in df.iterrows():
            if row["label_bias"] == "No agreement":
                pass
            else:
                sub_dict = {"text": row["text"]}
                if row["label_bias"] == "Biased":
                    sub_dict["label"] = 1
                elif row["label_bias"] == "Non-biased":
                    sub_dict["label"] = 0
                lst.append(sub_dict)
        return lst


data = DataLoad.read_babe()

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)


def compute_metrics(p):
    pred_flat = np.argmax(p.predictions, axis=1).flatten()
    labels_flat = p.label_ids.flatten()
    return {"f1": f1_score(labels_flat, pred_flat)}


def tokenizing_function(data) -> list:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    # Tokenizing
    tokenized = []
    for i in range(len(data)):
        token = tokenizer(
            data[i]["text"],
            padding="max_length",
            truncation=True,
            max_length=40,
        )
        token["labels"] = data[i]["label"]
        tokenized.append(token)
    ten = []
    for i in range(len(tokenized)):
        x = {}
        for j in tokenized[i].keys():
            x[j] = torch.tensor(tokenized[i][j])
        ten.append(x)
    return ten


def training(train_data, val_tokenized):
    # Tokenize your data
    train_tokenized = tokenizing_function(train_data)

    # Initialize DataLoader
    train_dataloader = DataLoader(train_tokenized, batch_size=16, shuffle=True)

    # Initialize the RoBERTa model and the tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Training Arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_dir="./logs",
        output_dir="/lbm_finetuned_model",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()
    return trainer


def report(val_tokenized, trainer):
    # Run prediction
    predictions = trainer.predict(val_tokenized)

    # Get predicted labels
    pred_labels = np.argmax(predictions.predictions, axis=1)

    # Get true labels
    true_labels = predictions.label_ids

    # Print Classification Report
    print(classification_report(true_labels, pred_labels))


import spacy


class EntityRemoval:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def remove_and_count_named_entities(self, text):
        doc = self.nlp(text)
        removed_entities = [
            token.text for token in doc if token.ent_type_ != ""
        ]
        tokens = [token.text if token.ent_type_ == "" else "" for token in doc]
        return " ".join(tokens), len(removed_entities)

    def compile_dataset(self, data):
        data_without_entities = []
        for item in data:
            (
                modified_text,
                named_entities_removed,
            ) = self.remove_and_count_named_entities(
                item["text"],
            )
            item_copy = item.copy()
            item_copy["text"] = modified_text
            item_copy["named_entities_removed"] = named_entities_removed
            data_without_entities.append(item_copy)
        return data_without_entities



val_tokenized = tokenizing_function(val_data)
trainer = training(train_data, val_tokenized)
report(val_tokenized, trainer)

er = EntityRemoval()

val_data_without_entities = er.compile_dataset(val_data)
removed_tokenized = tokenizing_function(val_data_without_entities)
report(removed_tokenized, trainer)

rm_train_data = er.compile_dataset(train_data)
rm_trainer = training(rm_train_data, removed_tokenized)
report(removed_tokenized, rm_trainer)
report(val_tokenized, rm_trainer)
