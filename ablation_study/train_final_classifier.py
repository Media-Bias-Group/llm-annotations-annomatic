# %% imports
import logging
import warnings

import pandas as pd
import torch
import transformers
import wandb
from datasets import Dataset, load_dataset
from evaluation.utils import compute_metrics, compute_metrics_hf, set_random_seed
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# %% warning surpress
transformers.logging.set_verbosity(transformers.logging.ERROR)
logging.disable(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


# %% definitions
base_model = "roberta-base"
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# %% prepare model & data
train = pd.read_parquet('ablation_study/data/anno-lexical-train-balanced.parquet')
dev = pd.read_parquet('ablation_study/data/anno-lexical-dev-balanced.parquet')
test = pd.read_parquet('annomatic-dataset/data/training/anno-lexical-test.parquet')
babe_test = load_dataset("mediabiasgroup/BABE")["test"] # load public version of BABE dataset
basil_test = load_dataset("horychtom/BASIL")["train"] # load public version of BASIL dataset

# prep model
model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% tokenize data
# train
tok = tokenizer(
    train['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
anno_lex_train_t = Dataset.from_dict(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": train['label'].tolist(),
    },
)
# dev
tok = tokenizer(
    dev['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
anno_lex_dev_t = Dataset.from_dict(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": dev['label'].tolist(),
    },
)

# test
tok_annolex = tokenizer(
    test['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
anno_lex_test_t = Dataset.from_dict(
    {
        "input_ids": tok_annolex["input_ids"],
        "attention_mask": tok_annolex["attention_mask"],
        "label": test['label'].tolist(),
    },
)

# test babe
tok_babe = tokenizer(
    babe_test['text'],
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
babe_test_t = Dataset.from_dict(
    {
        "input_ids": tok_babe["input_ids"],
        "attention_mask": tok_babe["attention_mask"],
        "label": babe_test['label'],
    },
)

# test basil
tok_basil = tokenizer(
    basil_test['text'],
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
basil_test_t = Dataset.from_dict(
    {
        "input_ids": tok_basil["input_ids"],
        "attention_mask": tok_basil["attention_mask"],
        "label": basil_test['label'],
    },
)



# %%
wandb.init(entity="anonymous", project="ablation_study")
set_random_seed()

training_args = TrainingArguments(
    report_to="wandb",
    output_dir="./checkpoints",
    per_device_eval_batch_size=128,
    per_device_train_batch_size=32,
    num_train_epochs=5,
    save_total_limit=5,
    evaluation_strategy="steps",
    logging_steps=50,
    eval_steps=10,
    save_steps=10,
    disable_tqdm=False,
    weight_decay=0.05,
    learning_rate=2e-5,
    run_name="balanced_",
    metric_for_best_model="eval_loss",
    save_strategy="steps",
    load_best_model_at_end=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=anno_lex_train_t,
    eval_dataset=anno_lex_dev_t,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_hf,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.0,
        ),
    ],
)


trainer.train()
model.push_to_hub("anonymous")

# test anno_lex
test_dataloader_annolex = DataLoader(
    anno_lex_test_t,
    batch_size=32,
    collate_fn=data_collator,
)
result_dict = compute_metrics(test_dataloader_annolex, model)
result_dict = {f"annolex_{key}": value for key, value in result_dict.items()}
wandb.log(result_dict)

# test babe
test_dataloader_babe = DataLoader(
    babe_test_t,
    batch_size=32,
    collate_fn=data_collator,
)
result_dict = compute_metrics(test_dataloader_babe, model)
result_dict = {f"babe_{key}": value for key, value in result_dict.items()}
wandb.log(result_dict)

# test basil
test_dataloader_basil = DataLoader(
    basil_test_t,
    batch_size=32,
    collate_fn=data_collator,
)
result_dict = compute_metrics(test_dataloader_basil, model)
result_dict = {f"basil_{key}": value for key, value in result_dict.items()}
wandb.log(result_dict)


wandb.finish()
