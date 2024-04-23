import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Check if multiple GPUs are available
if torch.cuda.device_count() < 2:
    raise RuntimeError("This script requires at least 2 GPUs.")

# Load the model and tokenizer
model_name = "mediabiasgroup/roberta_mtl_media_bias"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Wrap the model with DataParallel
model = torch.nn.DataParallel(model)
model.to("cuda")  # Move the model to GPU


# Function to predict bias for a batch of sentences
def predict_bias_batch(sentences, batch_size=32):
    max_probs = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        max_probs.extend(probs[:, 1].cpu().tolist())
    return max_probs


# Load the dataset
df = pd.read_parquet("/kaggle/input/bla-parquet/sentences.parquet")

# Apply the classifier to sentences in batches
df["bias_estimate"] = predict_bias_batch(df["sentence"].tolist())

# Print or inspect the updated DataFrame
print(df.head())
