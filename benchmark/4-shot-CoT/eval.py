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


# In[2]:


# load

# load results
df_falcon_7b = pd.read_csv(f"./data/falcon-7b-instruct.csv")
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


def find_first_occurrence(
    string,
    instruction="Instruction:",
    statement1="The answer is BIASED",
    statement2="The answer is NOT BIASED",
):
    index_instruction = string.find(instruction)
    index1 = string.find(statement1)
    index2 = string.find(statement2)

    if index1 == -1 and index2 == -1:
        return "Neither instruction nor statements found in the given string."
    elif index1 == -1 and index2 == -1:
        return "Neither statement found in the given string."
    elif index1 == -1:
        if index_instruction == -1 or index2 < index_instruction:
            return f"{statement2}"
        else:
            return "After Instruction"
    elif index2 == -1:
        if index_instruction == -1 or index1 < index_instruction:
            return f"{statement1}"
        else:
            return "After Instruction"
    elif index1 < index2:
        if index_instruction == -1 or index1 < index_instruction:
            return f"{statement1}"
        else:
            return "After Instruction"
    else:
        if index_instruction == -1 or index2 < index_instruction:
            return f"{statement2}"
        else:
            return "After Instruction"


# # Falcon 7B

# In[3]:


df_falcon_7b.query("label == '?'")


# In[4]:


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
    elif (
        row["response"].startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_falcon_7b["label"] = df_falcon_7b.apply(update_label, axis=1)
df_falcon_7b.query("label == '?'")


# In[5]:


df_falcon_7b.loc[[815, 822, 2321, 2864, 3779, 4009], "label"] = "BIASED"
df_falcon_7b.loc[
    [
        948,
        1275,
        2050,
        2732,
        2735,
        3430,
        3433,
        3837,
    ],
    "label",
] = "NOT BIASED"

# undefined
# df_falcon_7b.loc[4009]['response']
df_falcon_7b.query("label == '?'")


# In[6]:


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
    "F1-Score with Falcon 7b with (4 shot CoT): ",
    f1_score(ground_truth, falcon_7b_label),
)
print(
    "Precision with Falcon 7b with (4 shot CoT): ",
    precision_score(ground_truth, falcon_7b_label),
)
print(
    "Recall with Falcon 7b with (4 shot CoT): ",
    recall_score(ground_truth, falcon_7b_label),
)
print(
    "Accuracy with Falcon 7b with (4 shot CoT): ",
    accuracy_score(ground_truth, falcon_7b_label),
)


# # Flan UL2

# In[8]:


df_flan_ul2.query("label == '?'")


# In[9]:


def update_label(row):
    if row["response"].startswith("Not biased") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].startswith("Not Biased") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].startswith("Not BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is neutral")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence presents factual information")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is classified as NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_flan_ul2["label"] = df_flan_ul2.apply(update_label, axis=1)
df_flan_ul2.query("label == '?'")


# In[10]:


# manual assignment

df_flan_ul2.loc[[], "label"] = "BIASED"
df_flan_ul2.loc[[3443], "label"] = "NOT BIASED"

df_flan_ul2.loc[3443]["response"]


# In[11]:


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


# In[12]:


print(
    "F1-Score with Flan UL2 (4 shot CoT): ",
    f1_score(ground_truth, flan_ul2_label),
)
print(
    "Precision with Flan UL2 (4 shot CoT): ",
    precision_score(ground_truth, flan_ul2_label),
)
print(
    "Recall with Flan UL2 (4 shot CoT): ",
    recall_score(ground_truth, flan_ul2_label),
)
print(
    "Accuracy with Flan UL2 (4 shot CoT): ",
    accuracy_score(ground_truth, flan_ul2_label),
)


# # GPT-3.5-turbo

# In[13]:


df_openai_gpt_3_5_turbo.query("label == '?'")


# In[14]:


def update_label(row):
    if (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence presents factual")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence provides factual")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("This sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence above is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence is classified as NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence is considered not biased",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence is classified as not biased",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_openai_gpt_3_5_turbo["label"] = df_openai_gpt_3_5_turbo.apply(
    update_label,
    axis=1,
)
df_openai_gpt_3_5_turbo.query("label == '?'")


# In[15]:


# manual assignment
# if there is a tendency like 'may be biased/potential bias' or 'is factual' we classified it as biased or not biased respectively
#
# Rules:
# provided does not contain/exhibit any explicit bias -> NOT BIASED
#
df_openai_gpt_3_5_turbo.loc[[], "label"] = "BIASED"
df_openai_gpt_3_5_turbo.loc[
    [
        3898,
        3617,
        3612,
        3581,
        3124,
        3022,
        2748,
        2403,
        2341,
        2259,
        2126,
        2088,
        2006,
        1873,
        1806,
        1707,
        1617,
        1533,
        1478,
        1435,
        1391,
        1090,
        768,
        659,
        608,
        570,
        569,
        422,
        414,
        360,
        348,
        276,
        230,
        217,
        161,
    ],
    "label",
] = "NOT BIASED"

# NOT DETERMINABLE 3504, 3320, 3184, 3018, 2954, 2630, 1825, 1625, 1068,
#                   533, 413, 128

df_openai_gpt_3_5_turbo.loc[3184]["response"]
df_openai_gpt_3_5_turbo.query("label == '?'")


# In[16]:


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


# In[17]:


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


# # GPT 4 Turbo

# In[18]:


df_openai_gpt_4_turbo.query("label == '?'")


# In[19]:


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
    elif (
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_openai_gpt_4_turbo["label"] = df_openai_gpt_4_turbo.apply(
    update_label,
    axis=1,
)
df_openai_gpt_4_turbo.query("label == '?'")


# In[20]:


df_openai_gpt_4_turbo.loc[
    [
        3327,
        964,
        3057,
        3106,
        3163,
        2714,
        2486,
        2500,
        2159,
        2078,
        2021,
        1826,
        1386,
        1194,
    ],
    "label",
] = "BIASED"
df_openai_gpt_4_turbo.loc[
    [
        3787,
        3769,
        79,
        26,
        410,
        3571,
        360,
        3601,
        627,
        3184,
        3509,
        3557,
        911,
        936,
        3156,
        1080,
        1105,
        2983,
        2868,
        2777,
        2794,
        2516,
        2446,
        2399,
        2344,
        2192,
        2163,
        2051,
        1999,
        1950,
        1897,
        1895,
        1847,
        1720,
        1412,
        1251,
        1224,
        1128,
        1127,
    ],
    "label",
] = "NOT BIASED"

# nicht zuweisbar (oder context dependend)
df_openai_gpt_4_turbo.loc[
    [
        4019,
        3791,
        3766,
        128,
        81,
        75,
        246,
        364,
        381,
        427,
        3588,
        3620,
        3717,
        438,
        533,
        737,
        747,
        3499,
        809,
        907,
        3093,
        1041,
        1068,
        1117,
        2950,
        2954,
        2991,
        3018,
        2879,
        2865,
        2799,
        2694,
        2663,
        2630,
        2516,
        2433,
        2337,
        2217,
        2169,
        2117,
        2044,
        2040,
        1986,
        1975,
        1963,
        1914,
        1849,
        1825,
        1810,
        1612,
        1611,
        1580,
        1537,
        1529,
        1391,
        1348,
        1422,
        1244,
        1145,
    ],
    "label",
] = "!"

# 716...2879
# we use what GPT leans towards X as label

df_openai_gpt_4_turbo.loc[1127]["response"]
df_openai_gpt_4_turbo.query("label == '?'")


# In[21]:


df_openai_gpt_4_turbo["label"] = df_openai_gpt_4_turbo["label"].replace(
    "!",
    "?",
)

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


# In[22]:


print(
    "F1-Score with GPT 4 turbo with (4 shot CoT): ",
    f1_score(ground_truth, gpt_4_label),
)
print(
    "Precision with GPT 4 turbo with (4 shot CoT): ",
    precision_score(ground_truth, gpt_4_label),
)
print(
    "Recall with GPT 4 turbo with (4 shot CoT): ",
    recall_score(ground_truth, gpt_4_label),
)
print(
    "Accuracy with GPT 4 turbo with (4 shot CoT): ",
    accuracy_score(ground_truth, gpt_4_label),
)


# # Llama-2-7b-chat-hf

# In[23]:


df_Llama_2_7b.query("label == '?'")


# In[24]:


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
    elif (
        row["response"].startswith("The sentence is classified as NOT BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_Llama_2_7b["label"] = df_Llama_2_7b.apply(update_label, axis=1)
df_Llama_2_7b.query("label == '?'")


# In[25]:


df_Llama_2_7b.loc[[3624, 1269], "label"] = "BIASED"
df_Llama_2_7b.loc[
    [
        3545,
        3542,
        3491,
        3472,
        3373,
        3351,
        3227,
        3014,
        2830,
        2504,
        2346,
        2050,
        1828,
        1813,
        1639,
        1617,
        1551,
        1526,
        1287,
        1061,
        1029,
        921,
        593,
        241,
    ],
    "label",
] = "NOT BIASED"

df_Llama_2_7b.loc[241]["response"]
df_Llama_2_7b.query("label == '?'")


# In[26]:


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


# In[27]:


print(
    "F1-Score with llama 7b (4 shot CoT): ",
    f1_score(ground_truth, llama_7b_label),
)
print(
    "Precision with llama 7b (4 shot CoT): ",
    precision_score(ground_truth, llama_7b_label),
)
print(
    "Recall with llama 7b (4 shot CoT): ",
    recall_score(ground_truth, llama_7b_label),
)
print(
    "Accuracy with llama 7b (4 shot CoT): ",
    accuracy_score(ground_truth, llama_7b_label),
)


# # Llama-2-13b-chat-hf

# In[28]:


df_Llama_2_13b.query("label == '?'")


# In[29]:


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
    elif (
        row["response"].startswith("The sentence is classified as NOT BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_Llama_2_13b["label"] = df_Llama_2_13b.apply(update_label, axis=1)
df_Llama_2_13b.query("label == '?'")


# In[30]:


df_Llama_2_13b.loc[
    [3987, 2445, 1670, 1275, 1111, 656, 232],
    "label",
] = "BIASED"
df_Llama_2_13b.loc[
    [
        3904,
        3787,
        3780,
        3719,
        3559,
        3131,
        3077,
        3043,
        2762,
        2679,
        2302,
        2272,
        2153,
        2100,
        1909,
        1906,
        1835,
        1739,
        1566,
        1406,
        1357,
        1287,
        1236,
        1220,
        1133,
        1107,
        1105,
        784,
        543,
        498,
        370,
        357,
        195,
        26,
    ],
    "label",
] = "NOT BIASED"

# undefined
df_Llama_2_13b.loc[26]["response"]
df_Llama_2_13b.query("label == '?'")


# In[31]:


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


# In[32]:


print(
    "F1-Score with TODO with (4 shot CoT): ",
    f1_score(ground_truth, llama_13b_label),
)
print(
    "Precision with TODO with (4 shot CoT): ",
    precision_score(ground_truth, llama_13b_label),
)
print(
    "Recall with TODO with (4 shot CoT): ",
    recall_score(ground_truth, llama_13b_label),
)
print(
    "Accuracy with TODO with (4 shot CoT): ",
    accuracy_score(ground_truth, llama_13b_label),
)


# # Mistral-7B-Instruct-v0.1

# In[33]:


df_mistral_7b.query("label == '?'")


# In[34]:


# preprocessing
def update_label(row):
    if (
        row["response"].startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The statement is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is labeled as biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is classified as biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is classified as BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "The sentence above is classified as biased",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("This sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence is considered not biased",
        )
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
            "The sentence is classified as NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence is classified as not biased",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is unbiased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is NOT biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is factual and neutral")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence is neutral and presents factual information",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is factual")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is factual")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_mistral_7b["label"] = df_mistral_7b.apply(update_label, axis=1)
df_mistral_7b.query("label == '?'")


# In[35]:


df_mistral_7b.loc[
    [
        25,
        33,
        159,
        244,
        355,
        418,
        440,
        493,
        639,
        678,
        760,
        770,
        808,
        865,
        901,
        908,
        925,
        999,
        1099,
        1107,
        1153,
        1228,
        1233,
        1283,
        1380,
        1424,
        1435,
        1461,
        1472,
        1516,
        1520,
        1532,
        1627,
        1641,
        1732,
        1789,
        1835,
        1860,
        1940,
        1979,
        1985,
        2025,
        2046,
        2074,
        2151,
        2226,
        2251,
        2319,
        2341,
        2359,
        2484,
        2534,
        2610,
        2644,
        2658,
        2660,
        2732,
        2742,
        2798,
        2835,
        2902,
        2922,
        2953,
        2954,
        3126,
        3143,
        3182,
        3281,
        3318,
        3348,
        3361,
        3426,
        3443,
        3557,
        3560,
        3565,
        3582,
        3601,
        3623,
        3694,
        3711,
        3755,
        3795,
        3801,
        3837,
        3859,
        3907,
        3940,
    ],
    "label",
] = "NOT BIASED"

df_mistral_7b.loc[[635, 692, 1451, 1559, 2181, 2970], "label"] = "BIASED"

# undecideable 2860

df_mistral_7b.query("label == '?'")


# In[36]:


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


# In[37]:


print(
    "F1-Score with Mistral-7B-Instruct-v0.1 with (4 shot CoT): ",
    f1_score(ground_truth, df_mistral_7b_label),
)
print(
    "Precision with Mistral-7B-Instruct-v0.1 with (4 shot CoT): ",
    precision_score(ground_truth, df_mistral_7b_label),
)
print(
    "Recall with Mistral-7B-Instruct-v0.1 with (4 shot CoT): ",
    recall_score(ground_truth, df_mistral_7b_label),
)
print(
    "Accuracy with Mistral-7B-Instruct-v0.1 with (4 shot CoT): ",
    accuracy_score(ground_truth, df_mistral_7b_label),
)


# # Mixtral-8x7B

# In[38]:


df_mixtral_8x7b.query("label == '?'")


# In[39]:


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
        row["response"].startswith("The statement is biased")
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
    elif (
        row["response"].startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "The sentence above is classified as NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is classified as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
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
        row["response"].startswith("The sentence is labeled as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence appears to be NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_mixtral_8x7b["label"] = df_mixtral_8x7b.apply(update_label, axis=1)
df_mixtral_8x7b.query("label == '?'")


# In[40]:


df_mixtral_8x7b.loc[[3526, 1111, 75], "label"] = "BIASED"
df_mixtral_8x7b.loc[
    [
        3610,
        3410,
        3188,
        3077,
        2954,
        2796,
        2163,
        2052,
        1779,
        1777,
        1424,
        1414,
        748,
        600,
        329,
    ],
    "label",
] = "NOT BIASED"

# undefined 2964, 2630, 29

df_mixtral_8x7b.loc[29]["response"]
df_mixtral_8x7b.query("label == '?'")


# In[41]:


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


# In[42]:


print(
    "F1-Score with mixtral_8x7b with (4 shot CoT): ",
    f1_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Precision with mixtral_8x7b with (4 shot CoT): ",
    precision_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Recall with mixtral_8x7b with (4 shot CoT): ",
    recall_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Accuracy with mixtral_8x7b with (4 shot CoT): ",
    accuracy_score(ground_truth, df_mixtral_8x7b_label),
)


# # OpenChat_3.5

# In[43]:


df_openchat_3_5.query("label == '?'")


# In[44]:


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
    elif (
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_openchat_3_5["label"] = df_openchat_3_5.apply(update_label, axis=1)
df_openchat_3_5.query("label == '?'")


# In[45]:


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


# In[46]:


print(
    "F1-Score with OpenChat 3.5 with (4 shot CoT): ",
    f1_score(ground_truth, openchat_label),
)
print(
    "Precision with OpenChat 3.5 with (4 shot CoT): ",
    precision_score(ground_truth, openchat_label),
)
print(
    "Recall with OpenChat 3.5 with (4 shot CoT): ",
    recall_score(ground_truth, openchat_label),
)
print(
    "Accuracy with OpenChat 3.5 with (4 shot CoT): ",
    accuracy_score(ground_truth, openchat_label),
)


# # zephyr-7b-beta

# In[47]:


df_zephyr_7b_beta.query("label == '?'")


# In[48]:


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
    elif (
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_zephyr_7b_beta["label"] = df_zephyr_7b_beta.apply(update_label, axis=1)
df_zephyr_7b_beta.query("label == '?'")


# In[49]:


df_zephyr_7b_beta.loc[[32, 941, 3830], "label"] = "BIASED"
df_zephyr_7b_beta.loc[
    [
        18,
        22,
        66,
        371,
        391,
        657,
        819,
        884,
        935,
        1283,
        1520,
        1627,
        1760,
        1795,
        1968,
        2006,
        2089,
        2393,
        2494,
        2645,
        2714,
        2727,
        2898,
        2936,
        3039,
        3188,
        3458,
        3659,
        3745,
        1978,
        1780,
        2009,
        2941,
        2954,
        3177,
    ],
    "label",
] = "NOT BIASED"

# undecideable 562, 1051

df_zephyr_7b_beta.loc[3177]["response"]
df_zephyr_7b_beta.query("label == '?'")


# In[50]:


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


# In[51]:


print(
    "F1-Score with zephyr beta (4 shot CoT): ",
    f1_score(ground_truth, zephyr_label),
)
print(
    "Precision with zephyr beta (4 shot CoT): ",
    precision_score(ground_truth, zephyr_label),
)
print(
    "Recall with zephyr beta (4 shot CoT): ",
    recall_score(ground_truth, zephyr_label),
)
print(
    "Accuracy with zephyr beta (4 shot CoT): ",
    accuracy_score(ground_truth, zephyr_label),
)


# In[52]:


# safe the file
df_merge_all_runs_with_errors.to_csv("./all_runs_with_errors.csv", index=False)
