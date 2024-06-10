# %% imports
import logging
import warnings

import pandas as pd
import torch
import transformers
import wandb
from datasets import Dataset, load_dataset
from coreset.utils import compute_metrics, compute_metrics_hf, set_random_seed, wandb_run
from coreset.config import HF_TOKEN
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split


# %% warning surpress
transformers.logging.set_verbosity(transformers.logging.ERROR)
logging.disable(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


# %% definitions
base_model = "roberta-base"
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def tokenize_data(df, tokenizer):
    tok = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    return Dataset.from_dict(
        {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "label": df['label'].tolist(),
        }
    )

def evaluate_model(dataset, dataset_name, model, data_collator):
    test_dataloader = DataLoader(dataset, batch_size=32, collate_fn=data_collator)
    result_dict = compute_metrics(test_dataloader, model)
    result_dict = {f"{dataset_name}_{key}": value for key, value in result_dict.items()}
    wandb.log(result_dict)


def main(run_name):

    if run_name == "coreset-small":
        dataset = load_dataset("mandlc/anno-lexical-coreset-small-cluster")
    elif run_name == "coreset-big":
        dataset = load_dataset("mandlc/anno-lexical-coreset-big-cluster")
    elif run_name == "coreset-small-train":
        dataset = load_dataset("mandlc/anno-lexical-train-coreset-small-cluster")
    elif run_name == "coreset-big-train":
        dataset = load_dataset("mandlc/anno-lexical-train-coreset-big-cluster")
    

    train = dataset['train'].to_pandas()

    train, dev = train_test_split(
        train,
        test_size=0.2,
        stratify=train['label'],
        random_state=42,
    )

    test = dataset['test'].to_pandas()
    babe_test = load_dataset("mediabiasgroup/BABE")["test"]
    basil_test = load_dataset("horychtom/BASIL")["train"]

    # prep model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # %% tokenize data

    train_t = tokenize_data(train, tokenizer)
    dev_t = tokenize_data(dev, tokenizer)
    test_t = tokenize_data(test, tokenizer)
    
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
            "label": basil_test['lex_label'],
        },
    )
    # %%
    wandb.init(entity="media-bias-group", project="coreset_classifier", name=run_name)
    set_random_seed()
    from huggingface_hub import login
    login(HF_TOKEN)

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
        train_dataset=train_t,
        eval_dataset=dev_t,
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
    model.push_to_hub(f"mandlc/{run_name}")

    evaluate_model(test_t, "test_set", model, data_collator)
    evaluate_model(babe_test_t, "babe", model, data_collator)
    evaluate_model(basil_test_t, "basil", model, data_collator)

    wandb.finish()


main("coreset-small")
main("coreset-big")
main("coreset-small-train")
main("coreset-big-train")