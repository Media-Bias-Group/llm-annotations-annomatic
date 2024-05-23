import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


# INV 01
file_path = "checklist/data/spurious_cues_templates.xlsx"
xls = pd.ExcelFile(file_path)
dataframes = {}
for sheet_name in xls.sheet_names:
    dataframes[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)

model_name = "mediabiasgroup/babe-base-annomatic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

f1_scores = []

for bias in tqdm(dataframes["Overview"]["Category"]):
    row = dataframes["Overview"][dataframes["Overview"]["Category"] == bias]
    test_sentences = [
        row["Sentence biased 1"].values[0],
        row["Sentence biased 2"].values[0],
        row["Sentence neutral 1"].values[0],
        row["Sentence neutral 2"].values[0],
    ]
    final_sentences = []
    for i, sentence in enumerate(test_sentences):
        if i < 2:
            label = 1
        else:
            label = 0
        for category in dataframes[bias].columns:
            words = dataframes[bias][category]
            words = list(
                filter(lambda x: x is not None and not pd.isna(x), words),
            )
            for word in words:
                replaced_sentence = sentence.replace("[" + bias + "]", word)
                final_sentences.append([replaced_sentence, label, category])
    model_predictions = []
    for sentence, ground_truth, category in final_sentences:
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Make a prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted label (0 for unbiased, 1 for biased)
        predicted_label = np.argmax(logits, axis=1).item()
        model_predictions.append(
            (sentence, predicted_label, ground_truth, category),
        )

    f1_cat = f1_score(
        [ele[2] for ele in model_predictions],
        [ele[1] for ele in model_predictions],
    )
    evaluation_metrics = {}

    for category in dataframes[bias].columns:
        category_predictions, ground_truth_labels, predicted_labels = (
            [],
            [],
            [],
        )
        for sentence, predicted_label, ground_truth, cat in model_predictions:
            if category == cat:
                category_predictions.append(
                    [sentence, predicted_label, ground_truth, cat],
                )
                ground_truth_labels.append(ground_truth)
                predicted_labels.append(predicted_label)
        accuracy = accuracy_score(ground_truth_labels, predicted_labels)
        precision = precision_score(ground_truth_labels, predicted_labels)
        recall = recall_score(ground_truth_labels, predicted_labels)
        f1 = f1_score(ground_truth_labels, predicted_labels)
        evaluation_metrics[category] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }

        f1_scores.append(f1)


print(np.mean(f1_scores))
    # Print the evaluation metrics
    # for category, metrics in evaluation_metrics.items():
    #     print(f"Bias: {bias}")
    #     print(f"Subcategory: {category}")
    #     print(f"Accuracy: {metrics['Accuracy']:.2f}")
    #     print(f"Precision: {metrics['Precision']:.2f}")
    #     print(f"Recall: {metrics['Recall']:.2f}")
    #     print(f"F1 Score: {metrics['F1 Score']:.2f}")
    #     print("\n")
