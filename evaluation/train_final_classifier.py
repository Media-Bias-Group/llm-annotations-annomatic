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
magpie = "mediabiasgroup/lbm_without_media_bias_pretrained"
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# %% prepare model & data
anno_lex = load_dataset("anonymous/anno-lexical")


model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% tokenize data


tok = tokenizer(
    list(anno_lex["train"]["text"]),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
anno_lex_train_t = Dataset.from_dict(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": anno_lex["train"]["label"],
    },
)

tok = tokenizer(
    list(anno_lex["dev"]["text"]),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
anno_lex_dev_t = Dataset.from_dict(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": anno_lex["dev"]["label"],
    },
)

tok = tokenizer(
    list(anno_lex["test"]["text"]),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
anno_lex_test_t = Dataset.from_dict(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": anno_lex["test"]["label"],
    },
)


# %%
wandb.init(entity="anonymous", project="annomatic_dataset")
# set_random_seed()

training_args = TrainingArguments(
    report_to="wandb",
    output_dir="./checkpoints",
    per_device_eval_batch_size=32,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    save_total_limit=3,
    evaluation_strategy="steps",
    logging_steps=50,
    eval_steps=50,
    save_steps=50,
    disable_tqdm=False,
    weight_decay=0.05,
    learning_rate=2e-5,
    run_name="annomatic_test_eval",
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
# %%
eval_dataloader = DataLoader(
    anno_lex_test_t,
    batch_size=32,
    collate_fn=data_collator,
)
print(compute_metrics(eval_dataloader, model))

model.push_to_hub("mediabiasgroup/roberta-anno-lexical")

wandb.log(compute_metrics(eval_dataloader, model))
wandb.finish()
