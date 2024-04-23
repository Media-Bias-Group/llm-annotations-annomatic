import itertools
import math
import warnings

import numpy as np
import pandas as pd
from krippendorff import alpha
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", "pandas")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


path = "8-shot-CoT"
benchmark = pd.read_csv("benchmark_results.csv")
df = pd.read_csv(f"{path}/all_runs_with_errors.csv")
cols = [
    col for col in df.columns if ("label" in col) and col != "label_opinion"
]
df = df[cols]

# split the following list into 2
name_by_label = dict(
    [
        ("falcon_7b_label", "Falcon-7B-Instruct "),
        ("flan_t5_xl_label", "FLAN-T5-XL "),
        ("flan_t5_base_label", "FLAN-T5-Base"),
        ("flan_t5_large_label", "FLAN-T5-Large"),
        ("flan_ul2_label", "Flan-UL2"),
        ("gpt_3_5_label", "GPT-3.5 Turbo"),
        ("gpt_4_label", "GPT-4 Turbo"),
        ("llama_7b_label", "LLama 2 7B Chat"),
        ("llama_13b_label", "LLama 2 13B Chat"),
        ("mistral_7b_label", "Mistral-7B-v0.1 Instruct"),
        ("mixtral_8x7b_label", "Mixtral-8x7B Instruct"),
        ("openchat_label", "OpenChat 3.5"),
        ("zephyr_label", "Zephyr 7B β"),
    ],
)
runs = list(name_by_label.keys())


def fix_nans(dataframe, missing_data="?"):
    data_subset = dataframe.replace(missing_data, np.nan)

    # cast all columns to int except np.nan
    for col in df.columns:
        data_subset[col] = pd.to_numeric(data_subset[col])

    return data_subset


df = fix_nans(df)


def compute_krippendorff_alpha(columns):
    data_list = columns.T.values.tolist()

    # Calculate Krippendorff's alpha
    alpha_value = alpha(reliability_data=data_list)

    return alpha_value


def compute_krippendorff_alpha_for_k_runs(df, runs, k=None):
    rowlist = []

    def compute_metrics(cols):
        viable_mask = ~cols.isna().all(axis=1)
        y_true = df.label[viable_mask]
        y_pred = cols[viable_mask].mode(axis=1)[0]

        return (
            f1_score(y_true=y_true, y_pred=y_pred),
            fbeta_score(y_true=y_true, y_pred=y_pred, beta=0.5),
            mcc(y_true=y_true, y_pred=y_pred),
            precision_score(y_true=y_true, y_pred=y_pred),
            recall_score(y_true=y_true, y_pred=y_pred),
        )

    for combination in tqdm(
        itertools.combinations(runs, k),
        total=math.comb(len(runs), k),
    ):
        # for combination in itertools.combinations(runs,k):

        combination = list(combination)
        alpha_value = (
            compute_krippendorff_alpha(df[combination]) if k != 1 else 0
        )

        f1_, fbeta_, mcc_, p_, r_ = compute_metrics(df[combination])

        rowlist.append(
            {
                "combination": combination,
                "alpha": alpha_value,
                "mcc": mcc_,
                "f1": f1_,
                "fbeta": fbeta_,
                "p": p_,
                "r": r_,
            },
        )
    print(f"{k} finished.")
    return pd.DataFrame(rowlist)


import concurrent.futures

# exclude runs that are not in the dataframe
runs = [run for run in tqdm(runs) if run in df.columns]

futures = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    for k in range(1, len(runs) + 1, 2):
        future = executor.submit(
            compute_krippendorff_alpha_for_k_runs,
            df,
            runs,
            k,
        )
        futures.append(future)


# Wait for all futures to complete and collect the results
results = [future.result() for future in futures]
import os

os.mkdir(f"agreement_analysis/{path}")
for result in results:
    if len(result) != 0:
        result.to_csv(
            f"agreement_analysis/{path}/{len(result['combination'][0])}.csv",
        )


combined = pd.concat(results)
combined["no_models"] = combined.combination.apply(lambda x: len(list(x)))
combined.to_csv(f"agreement_analysis/{path}/all.csv")
