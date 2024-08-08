# %% imports
import logging
import warnings

import pandas as pd
import torch
import transformers
import wandb
from datasets import Dataset, load_dataset
from evaluation.utils import compute_metrics, set_random_seed, wandb_run
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# %% logging surpress
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
babe = load_dataset("mediabiasgroup/BABE-v3")["train"].to_pandas()

pool = pd.read_csv("../data/pool/final_pool_with_explanations.csv")
babe = (
    babe.merge(
        pool["text"],
        on="text",
        how="left",
        indicator=True,
    )
    .query(
        '_merge == "left_only"',
    )
    .drop("_merge", axis=1)
)

anno_lex = load_dataset("anonymous/anno-lexical")


model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% tokenize data
tok = tokenizer(
    list(babe["text"]),
    truncation=True,
    padding=True,
    max_length=128,
)
babe_t = pd.DataFrame(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": babe["label"],
    },
)

tok = tokenizer(
    list(anno_lex["train"]["text"]),
    truncation=True,
    padding=True,
    max_length=128,
)
anno_lex_train_t = pd.DataFrame(
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
)
anno_lex_dev_t = pd.DataFrame(
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
)
anno_lex_test_t = pd.DataFrame(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": anno_lex["test"]["label"],
    },
)

# %% Training
training_args = TrainingArguments(
    output_dir="../data/training/training_cache/",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    logging_steps=50,
    disable_tqdm=False,
    save_total_limit=0,
    weight_decay=0.1,
    warmup_ratio=0.1,
    learning_rate=4e-5,
)


# %% define cv
@wandb_run()
def run_cv(run_name):
    scores = []
    for train_index, val_index in skfold.split(
        babe_t["input_ids"],
        babe["label"],
    ):
        train_t = (
            Dataset.from_dict(babe_t.iloc[train_index])
            if run_name.split("-")[0] == "baseline"
            else Dataset.from_pandas(anno_lex_train_t)
        )
        dev_t = Dataset.from_dict(babe_t.iloc[val_index])
        set_random_seed()
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=2,
        )
        model.to(device)
        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_t,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        trainer.train()

        # evaluation
        eval_dataloader = DataLoader(
            dev_t,
            batch_size=32,
            collate_fn=data_collator,
        )
        scores.append(compute_metrics(eval_dataloader, model))
        wandb.log(scores[-1])

    df = pd.DataFrame(scores)
    final_values = df.mean().to_dict()
    wandb.log({"final_" + k: v for k, v in final_values.items()})


# %%
run_cv(run_name="baseline")
run_cv(run_name="anno-lexical")
