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

file_path = "checklist/data/quotation_DIR.xlsx"
df_q = pd.read_excel(file_path)

model_name = "mediabiasgroup/roberta-anno-lexical-ft"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model_predictions = []
for i in df_q.iterrows():
    inputs = tokenizer(
        i[1]["Biased Sentence"],
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
        (
            i[1]["Biased Sentence"],
            predicted_label,
            i[1]["Label"],
            i[1]["Category"],
        ),
    )

evaluation_metrics = {}
for i in range(0, len(model_predictions), 50):
    chunk = model_predictions[i : i + 50]
    ground_truth_labels = [ele[2] for ele in chunk]
    predicted_labels = [ele[1] for ele in chunk]
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    precision = precision_score(ground_truth_labels, predicted_labels)
    recall = recall_score(ground_truth_labels, predicted_labels)
    f1 = f1_score(ground_truth_labels, predicted_labels)
    evaluation_metrics[i] = {
        "Range": i,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }

# Print the evaluation metrics
for category, metrics in evaluation_metrics.items():
    print(f"Range: {metrics['Range']}")
    print(f"Accuracy: {metrics['Accuracy']:.2f}")
    print(f"Precision: {metrics['Precision']:.2f}")
    print(f"Recall: {metrics['Recall']:.2f}")
    print(f"F1 Score: {metrics['F1 Score']:.2f}")
    print("\n")
