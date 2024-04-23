# %%
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from fuzzywuzzy import fuzz


def fuzzify(text):
    return fuzz._process_and_sort(text, force_ascii=True)


# %%
babe = load_dataset("mediabiasgroup/BABE-v3")["train"].to_pandas()

d1 = pd.read_csv("mismatch_reann/timo_.csv")[:-1]
d2 = pd.read_csv("mismatch_reann/christoph_.csv")
d3 = pd.read_csv("mismatch_reann/tomas_.csv")[:-1]

reann = pd.DataFrame(
    {
        "text": d1.text,
        "timo": d1.Timo,
        "christoph": d2.Christoph,
        "tomas": d3.Tomas,
        "gpt": d1.gpt_4_label,
        "babe": d1.label,
    },
)
reann["combined"] = reann[["timo", "tomas", "christoph"]].mode(axis=1)[0]
pool = pd.read_csv("final_pool_with_explanations.csv")


# %%
babe["fuzz"] = babe.text.apply(fuzzify)
reann["fuzz"] = reann.text.apply(fuzzify)
pool["fuzz"] = pool.text.apply(fuzzify)

pool["label"] = pool.label.apply(lambda x: 1 if x == "BIASED" else 0)
pool = pool.rename(columns={"label": "new_label"})

# %% Merge reannotations where GPT-4 and babe disagreed
merged = babe.merge(reann, how="left", on="fuzz")
merged["label"].update(merged["combined"])
merged.drop_duplicates("text_x")

babe = merged[
    [
        "text_x",
        "news_link",
        "outlet",
        "topic",
        "type",
        "label",
        "label_opinion",
        "biased_words",
        "fuzz",
    ]
].rename(columns={"text_x": "text"})


# %% Merge reannotations on 100 pool sentences
merged = babe.merge(pool, how="left", on="fuzz")
merged["label"].update(merged["new_label"])


# %%
babe = merged[
    [
        "text_x",
        "news_link",
        "outlet",
        "topic",
        "type",
        "label",
        "label_opinion",
        "biased_words",
    ]
].rename(columns={"text_x": "text"})

# %%
babe.to_csv("babe-v4.csv", index=False)
