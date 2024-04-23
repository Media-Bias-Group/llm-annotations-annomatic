# %%
import huggingface_hub
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

babe = load_dataset("mediabiasgroup/BABE-v4")["train"].to_pandas()
pool = load_dataset("mediabiasgroup/BABE-icl-pool")["train"].to_pandas()

babe_t = babe[~babe["uuid"].isin(pool["uuid"])]


train, test = train_test_split(
    babe_t,
    train_size=3021,
    test_size=1000,
    stratify=babe_t["label"],
    random_state=42,
)
train = pd.concat([train, pool]).drop(columns=["explanation"])
train["label"] = train.label.astype(test.label.dtype)

ds = DatasetDict(
    {
        "train": Dataset.from_pandas(train, preserve_index=False),
        "test": Dataset.from_pandas(test, preserve_index=False),
    },
)

# v3
# ds.push_to_hub(
#     "mediabiasgroup/BABE",
#     commit_message="Upload version of BABE v2 split intro train and test",
#     commit_description="This version is the same as version v2, just split into train and test splits. The dev split is left for the user to choose.",
# )
# huggingface_hub.create_tag("mediabiasgroup/BABE", tag="v3", repo_type="dataset")
