#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import re
from typing import List, Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def is_contained(inner_start, inner_end, outer_start, outer_end):
    return (
        outer_start <= inner_start <= outer_end
        and outer_start <= inner_end <= outer_end
        and (inner_start, inner_end) != (outer_start, outer_end)
    )


def find_all_occurrences_indices(
    sentence: str,
    label: str,
) -> List[Tuple[int, int]]:
    """
    Finds all occurrences of a label in a sentence and returns a list of tuples
    with the start and end indices of the occurrences.

    Args:
        sentence: The sentence to search in.
        label: The label to search for.

    Returns:
        A list of tuples with the start and end indices of the occurrences.
    """
    return [
        (int(match.start()), int(match.end()))
        for match in re.finditer(label, sentence)
    ]


def find_contained_labels(labels):
    """
    Finds all labels that are contained in other labels.

    Args:
        labels: A list of labels.

    Returns:
        A dictionary with the labels as keys and a list of labels that are
        contained in the key label as values.
    """
    return {
        label_1: [
            label2
            for label2 in labels
            if label_1 in label2 and label_1 != label2
        ]
        for label_1 in labels
    }


def find_labels_in_sentence(
    sentence: str,
    labels: List[str],
) -> List[List[Tuple[int, int]]]:
    """
    Finds all occurrences of the labels in a sentence and returns a list of
    lists with the occurrences of each label.

    Labels that are contained in other labels are not returned.

    Args:
        sentence: The sentence to search in.
        labels: The labels to search for.

    Returns:
        A list of lists with the positional occurrence of each label.
    """
    if not labels:
        return []

    containable_map = find_contained_labels(labels)
    sentence_lower = sentence.lower()
    occurrences = [
        find_all_occurrences_indices(sentence_lower, label.lower())
        for label in labels
    ]
    label_occurrences = dict(zip(labels, occurrences))
    included_list = []
    for label, inner_pos in zip(labels, occurrences):
        contained = set()
        containing_labels = containable_map.get(label, set())

        for bigger_label in containing_labels:
            locations: List[Tuple[int, int]] = label_occurrences.get(
                bigger_label,
                list(),
            )

            for inner_start, inner_end in inner_pos:
                for outer_start, outer_end in locations:
                    if is_contained(
                        inner_start,
                        inner_end,
                        outer_start,
                        outer_end,
                    ):
                        contained.add((inner_start, inner_end))
        included_list.append(list(contained))

    return [
        list(set(pos) - set(included))
        for pos, included in zip(occurrences, included_list)
    ]


def find_label(
    sentence: str,
    labels: List[str],
    default_label: str = "?",
) -> str:
    """
    Search for given labels in the sentence and returns it if found. If only
    one label occur in the sentence, it will be returned. If no label or
    different labels occur in the sentence, '?' is returned.

    Args:
        sentence: The sentence to search in.
        labels: The labels to search for.
        default_label: The label to return if no label or different labels
            occur in the sentence.

    Returns:
        The label that occurs in the sentence or '?' if no label occurs in the
        sentence.
    """
    occurrences = find_labels_in_sentence(sentence=sentence, labels=labels)
    non_empty_indices = [i for i, sublist in enumerate(occurrences) if sublist]
    return (
        labels[non_empty_indices[0]]
        if len(
            non_empty_indices,
        )
        == 1
        else default_label
    )


def _soft_parse(
    df: pd.DataFrame,
    in_col: str,
    parsed_col: str,
    labels: List[str] = None,
) -> pd.DataFrame:
    if labels is None:
        raise ValueError("Labels are not set!")

    df[parsed_col] = df[in_col].apply(
        lambda x: find_label(x, labels),
    )


# In[4]:


# load

# load results
df_falcon_7b = pd.read_csv(f"./data/falcon-7b-instruct.csv")
df_flan_t5_base = pd.read_csv("./data/flan-t5-base.csv")
df_flan_t5_large = pd.read_csv("./data/flan-t5-large.csv")
df_flan_t5_xl = pd.read_csv("./data/flan-t5-xl.csv")
df_flan_ul2 = pd.read_csv(f"./data/flan-ul2.csv")
df_openai_gpt_3_5_turbo = pd.read_csv("./data/gpt-3.5-turbo.csv")
df_openai_gpt_4_turbo = pd.read_csv(f"./data/gpt-4-1106-preview.csv")
df_Llama_2_7b = pd.read_csv(f"./data/Llama-2-7b-chat-hf.csv")
df_Llama_2_13b = pd.read_csv(f"./data/Llama-2-13b-chat-hf.csv")
df_mistral_7b = pd.read_csv(f"./data/Mistral-7B-Instruct-v0.1.csv")
df_mixtral_8x7b = pd.read_csv(f"./data/Mixtral-8x7B-Instruct-v0.1.csv")
df_openchat_3_5 = pd.read_csv(f"./data/openchat_3.5.csv")
df_zephyr_7b_beta = pd.read_csv(f"./data/zephyr-7b-beta.csv")

# load pool
pool = load_dataset("mediabiasgroup/BABE-icl-pool")["train"].to_pandas()

# exclude pool from model (if needed)
df_falcon_7b = (
    df_falcon_7b.merge(
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

df_flan_t5_base = (
    df_flan_t5_base.merge(
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

df_flan_t5_large = (
    df_flan_t5_large.merge(
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

df_flan_t5_xl = (
    df_flan_t5_xl.merge(
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

df_flan_ul2 = (
    df_flan_ul2.merge(
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
df_openai_gpt_3_5_turbo = (
    df_openai_gpt_3_5_turbo.merge(
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
df_openai_gpt_4_turbo = (
    df_openai_gpt_4_turbo.merge(
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
df_Llama_2_7b = (
    df_Llama_2_7b.merge(
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
df_Llama_2_13b = (
    df_Llama_2_13b.merge(
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
df_mistral_7b = (
    df_mistral_7b.merge(
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
df_mixtral_8x7b = (
    df_mixtral_8x7b.merge(
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
df_openchat_3_5 = (
    df_openchat_3_5.merge(
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
df_zephyr_7b_beta = (
    df_zephyr_7b_beta.merge(
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


# load babe
dataset = load_dataset("mediabiasgroup/BABE")
df_babe = pd.DataFrame(dataset["train"])

# df_merge_all_runs = only contains the elements legal in all annotations
df_merge_all_runs = df_babe

# df_merge_all_runs_with_errors = only contains the elements legal in all annotations
df_merge_all_runs_with_errors = df_babe


# # Falcon 7B

# In[3]:


df_falcon_7b.query("label == '?'")


# In[4]:


# preprocessing
def update_label(row):
    if str(row["response"]).startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif str(row["response"]).startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif str(row["response"]).startswith("'BIASED'") and row["label"] == "?":
        return "BIASED"
    elif (
        str(row["response"]).startswith("'NOT BIASED'") and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        str(row["response"]).startswith("Classification: NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        str(row["response"]).startswith("Classification: BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        str(row["response"]).startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).startswith(
            "The sentence is classified as NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif pd.isna(row["response"]):
        return "!"
    else:
        return row["label"]


df_falcon_7b["label"] = df_falcon_7b.apply(update_label, axis=1)
df_falcon_7b.query("label == '?'")


# In[5]:


df_falcon_7b.loc[[87, 642, 1032, 1242, 1757, 3116], "label"] = "BIASED"
df_falcon_7b.loc[[1895, 2316, 3124, 3444, 3853], "label"] = "NOT BIASED"

df_falcon_7b.loc[[130], "label"] = "!"


# best answer: , which makes it NOT BIASED. The answer is correct.'

df_falcon_7b.query("label == '?'")


# In[6]:


# map "!" flag back to "?"
df_falcon_7b["label"] = df_falcon_7b["label"].replace("!", "?")

df_falcon_7b = df_falcon_7b.rename(columns={"label": "falcon_7b_label"})
df_falcon_7b["falcon_7b_label"] = df_falcon_7b["falcon_7b_label"].replace(
    "BIASED",
    1,
)
df_falcon_7b["falcon_7b_label"] = df_falcon_7b["falcon_7b_label"].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_falcon_7b[df_falcon_7b["falcon_7b_label"] != "?"][
        ["text", "falcon_7b_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_falcon_7b[df_falcon_7b["falcon_7b_label"] != "?"][
        ["text", "falcon_7b_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_falcon_7b[["text", "falcon_7b_label"]],
    on="text",
)

ground_truth = df_merge["label"].astype(int)
falcon_7b_label = df_merge["falcon_7b_label"].astype(int)


# In[7]:


print(
    "F1-Score with Falcon 7b with (2 shot): ",
    f1_score(ground_truth, falcon_7b_label),
)
print(
    "Precision with Falcon 7b with (2 shot): ",
    precision_score(ground_truth, falcon_7b_label),
)
print(
    "Recall with Falcon 7b with (2 shot): ",
    recall_score(ground_truth, falcon_7b_label),
)
print(
    "Accuracy with Falcon 7b with (2 shot): ",
    accuracy_score(ground_truth, falcon_7b_label),
)


# # Flan T5 Base

# In[8]:


df_flan_t5_base.query("label == '?'")


# In[9]:


df_flan_t5_base = df_flan_t5_base.rename(
    columns={"label": "flan_t5_base_label"},
)
df_flan_t5_base["flan_t5_base_label"] = df_flan_t5_base[
    "flan_t5_base_label"
].replace("BIASED", 1)
df_flan_t5_base["flan_t5_base_label"] = df_flan_t5_base[
    "flan_t5_base_label"
].replace("NOT BIASED", 0)

df_merge = df_babe.merge(
    df_flan_t5_base[df_flan_t5_base["flan_t5_base_label"] != "?"][
        ["text", "flan_t5_base_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_flan_t5_base[df_flan_t5_base["flan_t5_base_label"] != "?"][
        ["text", "flan_t5_base_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_flan_t5_base[["text", "flan_t5_base_label"]],
    on="text",
)

ground_truth = df_merge["label"].astype(int)
flan_t5_base_label = df_merge["flan_t5_base_label"].astype(int)


# In[10]:


print(
    "F1-Score with Flan T5 base (2 shot): ",
    f1_score(ground_truth, flan_t5_base_label),
)
print(
    "Precision with Flan T5 base (2 shot): ",
    precision_score(ground_truth, flan_t5_base_label),
)
print(
    "Recall with Flan T5 base (2 shot): ",
    recall_score(ground_truth, flan_t5_base_label),
)
print(
    "Accuracy with Flan T5 base (2 shot): ",
    accuracy_score(ground_truth, flan_t5_base_label),
)


# # Flan T5 Large

# In[11]:


df_flan_t5_large.query("label == '?'")


# In[12]:


df_flan_t5_large = df_flan_t5_large.rename(
    columns={"label": "flan_t5_large_label"},
)
df_flan_t5_large["flan_t5_large_label"] = df_flan_t5_large[
    "flan_t5_large_label"
].replace("BIASED", 1)
df_flan_t5_large["flan_t5_large_label"] = df_flan_t5_large[
    "flan_t5_large_label"
].replace("NOT BIASED", 0)

df_merge = df_babe.merge(
    df_flan_t5_large[df_flan_t5_large["flan_t5_large_label"] != "?"][
        ["text", "flan_t5_large_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_flan_t5_large[df_flan_t5_large["flan_t5_large_label"] != "?"][
        ["text", "flan_t5_large_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_flan_t5_large[["text", "flan_t5_large_label"]],
    on="text",
)

ground_truth = df_merge["label"].astype(int)
flan_t5_large_label = df_merge["flan_t5_large_label"].astype(int)


# In[13]:


print(
    "F1-Score with Flan T5 Large (2 shot): ",
    f1_score(ground_truth, flan_t5_large_label),
)
print(
    "Precision with Flan T5 Large (2 shot): ",
    precision_score(ground_truth, flan_t5_large_label),
)
print(
    "Recall with Flan T5 Large (2 shot): ",
    recall_score(ground_truth, flan_t5_large_label),
)
print(
    "Accuracy with Flan T5 Large (2 shot): ",
    accuracy_score(ground_truth, flan_t5_large_label),
)


# # Flan T5 XL

# In[14]:


df_flan_t5_xl.query("label == '?'")


# In[15]:


df_flan_t5_xl = df_flan_t5_xl.rename(columns={"label": "flan_t5_xl_label"})
df_flan_t5_xl["flan_t5_xl_label"] = df_flan_t5_xl["flan_t5_xl_label"].replace(
    "BIASED",
    1,
)
df_flan_t5_xl["flan_t5_xl_label"] = df_flan_t5_xl["flan_t5_xl_label"].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_flan_t5_xl[df_flan_t5_xl["flan_t5_xl_label"] != "?"][
        ["text", "flan_t5_xl_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_flan_t5_xl[df_flan_t5_xl["flan_t5_xl_label"] != "?"][
        ["text", "flan_t5_xl_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_flan_t5_xl[["text", "flan_t5_xl_label"]],
    on="text",
)

ground_truth = df_merge["label"].astype(int)
flan_t5_xl_label = df_merge["flan_t5_xl_label"].astype(int)


# In[16]:


print(
    "F1-Score with Flan T5 xl (2 shot): ",
    f1_score(ground_truth, flan_t5_xl_label),
)
print(
    "Precision with Flan T5 xl (2 shot): ",
    precision_score(ground_truth, flan_t5_xl_label),
)
print(
    "Recall with Flan T5 xl (2 shot): ",
    recall_score(ground_truth, flan_t5_xl_label),
)
print(
    "Accuracy with Flan T5 xl (2 shot): ",
    accuracy_score(ground_truth, flan_t5_xl_label),
)


# # Flan UL2

# In[17]:


df_flan_ul2.query("label == '?'")


# In[18]:


df_flan_ul2 = df_flan_ul2.rename(columns={"label": "flan_ul2_label"})
df_flan_ul2["flan_ul2_label"] = df_flan_ul2["flan_ul2_label"].replace(
    "BIASED",
    1,
)
df_flan_ul2["flan_ul2_label"] = df_flan_ul2["flan_ul2_label"].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_flan_ul2[df_flan_ul2["flan_ul2_label"] != "?"][
        ["text", "flan_ul2_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_flan_ul2[df_flan_ul2["flan_ul2_label"] != "?"][
        ["text", "flan_ul2_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_flan_ul2[["text", "flan_ul2_label"]],
    on="text",
)

ground_truth = df_merge["label"].astype(int)
flan_ul2_label = df_merge["flan_ul2_label"].astype(int)


# In[19]:


print(
    "F1-Score with Flan UL2 (2 shot): ",
    f1_score(ground_truth, flan_ul2_label),
)
print(
    "Precision with Flan UL2 (2 shot): ",
    precision_score(ground_truth, flan_ul2_label),
)
print(
    "Recall with Flan UL2 (2 shot): ",
    recall_score(ground_truth, flan_ul2_label),
)
print(
    "Accuracy with Flan UL2 (2 shot): ",
    accuracy_score(ground_truth, flan_ul2_label),
)


# # GPT-3.5-turbo

# In[20]:


df_openai_gpt_3_5_turbo.query("label == '?'")


# In[21]:


df_openai_gpt_3_5_turbo["label"] = df_openai_gpt_3_5_turbo.apply(
    update_label,
    axis=1,
)
df_openai_gpt_3_5_turbo.query("label == '?'")


# In[ ]:


# In[22]:


df_openai_gpt_3_5_turbo = df_openai_gpt_3_5_turbo.rename(
    columns={"label": "gpt_3_5_label"},
)
df_openai_gpt_3_5_turbo["gpt_3_5_label"] = df_openai_gpt_3_5_turbo[
    "gpt_3_5_label"
].replace(
    "BIASED",
    1,
)
df_openai_gpt_3_5_turbo["gpt_3_5_label"] = df_openai_gpt_3_5_turbo[
    "gpt_3_5_label"
].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_openai_gpt_3_5_turbo[df_openai_gpt_3_5_turbo["gpt_3_5_label"] != "?"][
        ["text", "gpt_3_5_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_openai_gpt_3_5_turbo[df_openai_gpt_3_5_turbo["gpt_3_5_label"] != "?"][
        ["text", "gpt_3_5_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_openai_gpt_3_5_turbo[["text", "gpt_3_5_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
gpt_3_5_label = df_merge["gpt_3_5_label"].astype(int)


# In[23]:


print(
    "F1-Score with GPT 3.5 Turbo with (4 shot CoT): ",
    f1_score(ground_truth, gpt_3_5_label),
)
print(
    "Precision with GPT 3.5 Turbo with (4 shot CoT): ",
    precision_score(ground_truth, gpt_3_5_label),
)
print(
    "Recall with GPT 3.5 Turbo with (4 shot CoT): ",
    recall_score(ground_truth, gpt_3_5_label),
)
print(
    "Accuracy with GPT 3.5 Turbo with (4 shot CoT): ",
    accuracy_score(ground_truth, gpt_3_5_label),
)


# # GPT 4 turbo

# In[24]:


df_openai_gpt_4_turbo.query("label == '?'")


# In[25]:


# preprocessing
def update_label(row):
    if row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        row["response"].startswith("Classification: NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("Classification: BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    else:
        return row["label"]


df_openai_gpt_4_turbo["label"] = df_openai_gpt_4_turbo.apply(
    update_label,
    axis=1,
)
df_openai_gpt_4_turbo.query("label == '?'")


# In[ ]:


# In[26]:


df_openai_gpt_4_turbo = df_openai_gpt_4_turbo.rename(
    columns={"label": "gpt_4_label"},
)
df_openai_gpt_4_turbo["gpt_4_label"] = df_openai_gpt_4_turbo[
    "gpt_4_label"
].replace(
    "BIASED",
    1,
)
df_openai_gpt_4_turbo["gpt_4_label"] = df_openai_gpt_4_turbo[
    "gpt_4_label"
].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_openai_gpt_4_turbo[df_openai_gpt_4_turbo["gpt_4_label"] != "?"][
        ["text", "gpt_4_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_openai_gpt_4_turbo[df_openai_gpt_4_turbo["gpt_4_label"] != "?"][
        ["text", "gpt_4_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_openai_gpt_4_turbo[["text", "gpt_4_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
gpt_4_label = df_merge["gpt_4_label"].astype(int)


# In[27]:


print(
    "F1-Score with GPT 4 turbo with (2 shot): ",
    f1_score(ground_truth, gpt_4_label),
)
print(
    "Precision with GPT 4 turbo with (2 shot): ",
    precision_score(ground_truth, gpt_4_label),
)
print(
    "Recall with GPT 4 turbo with (2 shot ): ",
    recall_score(ground_truth, gpt_4_label),
)
print(
    "Accuracy with GPT 4 turbo with (2 shot): ",
    accuracy_score(ground_truth, gpt_4_label),
)


# # Llama-2-7b-chat-hf

# In[28]:


df_Llama_2_7b.query("label == '?'")


# In[29]:


# preprocessing
def update_label(row):
    if row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    else:
        return row["label"]


df_Llama_2_7b["label"] = df_Llama_2_7b.apply(update_label, axis=1)
df_Llama_2_7b.query("label == '?'")


# In[ ]:


# In[30]:


df_Llama_2_7b = df_Llama_2_7b.rename(columns={"label": "llama_7b_label"})
df_Llama_2_7b["llama_7b_label"] = df_Llama_2_7b["llama_7b_label"].replace(
    "BIASED",
    1,
)
df_Llama_2_7b["llama_7b_label"] = df_Llama_2_7b["llama_7b_label"].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_Llama_2_7b[df_Llama_2_7b["llama_7b_label"] != "?"][
        ["text", "llama_7b_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_Llama_2_7b[df_Llama_2_7b["llama_7b_label"] != "?"][
        ["text", "llama_7b_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_Llama_2_7b[["text", "llama_7b_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
llama_7b_label = df_merge["llama_7b_label"].astype(int)


# In[31]:


print(
    "F1-Score with llama 7b (2 shot): ",
    f1_score(ground_truth, llama_7b_label),
)
print(
    "Precision with llama 7b (2 shot): ",
    precision_score(ground_truth, llama_7b_label),
)
print(
    "Recall with llama 7b (2 shot): ",
    recall_score(ground_truth, llama_7b_label),
)
print(
    "Accuracy with llama 7b (2 shot): ",
    accuracy_score(ground_truth, llama_7b_label),
)


# # Llama-2-13b-chat-hf

# In[32]:


df_Llama_2_13b.query("label == '?'")


# In[33]:


# preprocessing
def update_label(row):
    if row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    else:
        return row["label"]


df_Llama_2_13b["label"] = df_Llama_2_13b.apply(update_label, axis=1)
df_Llama_2_13b.query("label == '?'")


# In[ ]:


# In[34]:


df_Llama_2_13b = df_Llama_2_13b.rename(columns={"label": "llama_13b_label"})
df_Llama_2_13b["llama_13b_label"] = df_Llama_2_13b["llama_13b_label"].replace(
    "BIASED",
    1,
)
df_Llama_2_13b["llama_13b_label"] = df_Llama_2_13b["llama_13b_label"].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_Llama_2_13b[df_Llama_2_13b["llama_13b_label"] != "?"][
        ["text", "llama_13b_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_Llama_2_13b[df_Llama_2_13b["llama_13b_label"] != "?"][
        ["text", "llama_13b_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_Llama_2_13b[["text", "llama_13b_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
llama_13b_label = df_merge["llama_13b_label"].astype(int)


# In[35]:


print(
    "F1-Score with Llama 2 13b with (2 shot): ",
    f1_score(ground_truth, llama_13b_label),
)
print(
    "Precision with Llama 2 13b with (2 shot): ",
    precision_score(ground_truth, llama_13b_label),
)
print(
    "Recall with Llama 2 13b with (2 shot): ",
    recall_score(ground_truth, llama_13b_label),
)
print(
    "Accuracy with Llama 2 13b with (2 shot): ",
    accuracy_score(ground_truth, llama_13b_label),
)


# # Mistral-7B-Instruct-v0.1

# In[36]:


df_mistral_7b.query("label == '?'")


# In[37]:


# preprocessing
def update_label(row):
    if row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        row["response"].startswith("Classification: NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("Classification: BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif row["response"].startswith("Instruction:") and row["label"] == "?":
        return "!"
    else:
        return row["label"]


df_mistral_7b["label"] = df_mistral_7b.apply(update_label, axis=1)
df_mistral_7b.query("label == '?'")


# In[38]:


# Map back '!' flag to '?'
df_mistral_7b["label"] = df_mistral_7b["label"].replace("!", "?")

df_mistral_7b = df_mistral_7b.rename(columns={"label": "mistral_7b_label"})
df_mistral_7b["mistral_7b_label"] = df_mistral_7b["mistral_7b_label"].replace(
    "BIASED",
    1,
)
df_mistral_7b["mistral_7b_label"] = df_mistral_7b["mistral_7b_label"].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_mistral_7b[df_mistral_7b["mistral_7b_label"] != "?"][
        ["text", "mistral_7b_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_mistral_7b[df_mistral_7b["mistral_7b_label"] != "?"][
        ["text", "mistral_7b_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_mistral_7b[["text", "mistral_7b_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
df_mistral_7b_label = df_merge["mistral_7b_label"].astype(int)


# In[39]:


print(
    "F1-Score with Mistral-7B-Instruct-v0.1 with (2 shot): ",
    f1_score(ground_truth, df_mistral_7b_label),
)
print(
    "Precision with Mistral-7B-Instruct-v0.1 with (2 shot): ",
    precision_score(ground_truth, df_mistral_7b_label),
)
print(
    "Recall with Mistral-7B-Instruct-v0.1 with (2 shot): ",
    recall_score(ground_truth, df_mistral_7b_label),
)
print(
    "Accuracy with Mistral-7B-Instruct-v0.1 with (2 shot): ",
    accuracy_score(ground_truth, df_mistral_7b_label),
)


# # Mixtral-8x7B

# In[40]:


df_mixtral_8x7b.query("label == '?'")


# In[41]:


# preprocessing
# preprocessing
def update_label(row):
    if row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    else:
        return row["label"]


df_mixtral_8x7b["label"] = df_mixtral_8x7b.apply(update_label, axis=1)
df_mixtral_8x7b.query("label == '?'")


# In[42]:


df_mixtral_8x7b = df_mixtral_8x7b.rename(
    columns={"label": "mixtral_8x7b_label"},
)
df_mixtral_8x7b["mixtral_8x7b_label"] = df_mixtral_8x7b[
    "mixtral_8x7b_label"
].replace("BIASED", 1)
df_mixtral_8x7b["mixtral_8x7b_label"] = df_mixtral_8x7b[
    "mixtral_8x7b_label"
].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_mixtral_8x7b[df_mixtral_8x7b["mixtral_8x7b_label"] != "?"][
        ["text", "mixtral_8x7b_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_mixtral_8x7b[df_mixtral_8x7b["mixtral_8x7b_label"] != "?"][
        ["text", "mixtral_8x7b_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_mixtral_8x7b[["text", "mixtral_8x7b_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
df_mixtral_8x7b_label = df_merge["mixtral_8x7b_label"].astype(int)


# In[43]:


print(
    "F1-Score with mixtral_8x7b with (2 shot): ",
    f1_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Precision with mixtral_8x7b with (2 shot): ",
    precision_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Recall with mixtral_8x7b with (2 shot): ",
    recall_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Accuracy with mixtral_8x7b with (2 shot): ",
    accuracy_score(ground_truth, df_mixtral_8x7b_label),
)


# # OpenChat_3.5

# In[44]:


df_openchat_3_5.query("label == '?'")


# In[45]:


def update_label(row):
    if row["response"].startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].startswith("Not BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        row["response"].startswith("Answer: NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("Answer: BIASED") and row["label"] == "?":
        return "BIASED"
    elif (
        row["response"].startswith("Classification: BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence above is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence above is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "The sentence above is classified as BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("Classification: NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("Explanation: The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("Instruction: 'The media is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "Explanation: The sentence is a strong opinion and therefore biased.",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif row["response"].startswith("Instruction:") and row["label"] == "?":
        return "!"
    elif row["response"].startswith("Not enough") and row["label"] == "?":
        return "!"
    else:
        return row["label"]


df_openchat_3_5["label"] = df_openchat_3_5.apply(update_label, axis=1)
df_openchat_3_5.query("label == '?'")


# In[46]:


df_openchat_3_5.loc[[1631, 3358], "label"] = "BIASED"
df_openchat_3_5.loc[[2078, 2306], "label"] = "NOT BIASED"

df_openchat_3_5.loc[[2232, 3095, 3835], "label"] = "!"

df_openchat_3_5.loc[3358]["response"]
df_openchat_3_5.query("label == '?'")


# In[47]:


# Map back '!' flag to '?'
df_openchat_3_5["label"] = df_openchat_3_5["label"].replace("!", "?")

df_openchat_3_5 = df_openchat_3_5.rename(columns={"label": "openchat_label"})
df_openchat_3_5["openchat_label"] = df_openchat_3_5["openchat_label"].replace(
    "BIASED",
    1,
)
df_openchat_3_5["openchat_label"] = df_openchat_3_5["openchat_label"].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_openchat_3_5[df_openchat_3_5["openchat_label"] != "?"][
        ["text", "openchat_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_openchat_3_5[df_openchat_3_5["openchat_label"] != "?"][
        ["text", "openchat_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_openchat_3_5[["text", "openchat_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
openchat_label = df_merge["openchat_label"].astype(int)


# In[48]:


print(
    "F1-Score with OpenChat 3.5 with (2 shot): ",
    f1_score(ground_truth, openchat_label),
)
print(
    "Precision with OpenChat 3.5 with (2 shot): ",
    precision_score(ground_truth, openchat_label),
)
print(
    "Recall with OpenChat 3.5 with (2 shot): ",
    recall_score(ground_truth, openchat_label),
)
print(
    "Accuracy with OpenChat 3.5 with (2 shot): ",
    accuracy_score(ground_truth, openchat_label),
)


# # zephyr-7b-beta

# In[49]:


df_zephyr_7b_beta.query("label == '?'")


# In[50]:


# preprocessing
def update_label(row):
    if row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        row["response"].startswith("Classification: NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("Classification: BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    else:
        return row["label"]


df_zephyr_7b_beta["label"] = df_zephyr_7b_beta.apply(update_label, axis=1)
df_zephyr_7b_beta.query("label == '?'")


# In[51]:


df_zephyr_7b_beta = df_zephyr_7b_beta.rename(columns={"label": "zephyr_label"})
df_zephyr_7b_beta["zephyr_label"] = df_zephyr_7b_beta["zephyr_label"].replace(
    "BIASED",
    1,
)
df_zephyr_7b_beta["zephyr_label"] = df_zephyr_7b_beta["zephyr_label"].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_zephyr_7b_beta[df_zephyr_7b_beta["zephyr_label"] != "?"][
        ["text", "zephyr_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_zephyr_7b_beta[df_zephyr_7b_beta["zephyr_label"] != "?"][
        ["text", "zephyr_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_zephyr_7b_beta[["text", "zephyr_label"]],
    on="text",
)

ground_truth = df_merge["label"].astype(int)
zephyr_label = df_merge["zephyr_label"].astype(int)


# In[52]:


print(
    "F1-Score with zephyr beta (2 shot): ",
    f1_score(ground_truth, zephyr_label),
)
print(
    "Precision with zephyr beta (2 shot): ",
    precision_score(ground_truth, zephyr_label),
)
print(
    "Recall with zephyr beta (2 shot): ",
    recall_score(ground_truth, zephyr_label),
)
print(
    "Accuracy with zephyr beta (2 shot): ",
    accuracy_score(ground_truth, zephyr_label),
)


# In[53]:


# safe the file
df_merge_all_runs_with_errors.to_csv("./all_runs_with_errors.csv", index=False)
