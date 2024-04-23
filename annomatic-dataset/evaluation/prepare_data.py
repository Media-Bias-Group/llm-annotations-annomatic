# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
df = pd.read_parquet("../data/output/anno-lexical.parquet")

# %%
df = df[["text", "source_party", "source_name", "final_label", "sentence_id"]]
df = df.rename(columns={"final_label": "label"})

# %%
train_df, devtest_df = train_test_split(
    df, test_size=0.3, stratify=df.source_party, random_state=42
)
dev_df, test_df = train_test_split(
    devtest_df,
    test_size=0.5,
    stratify=devtest_df.source_party,
    random_state=42,
)


# %%
train_df.to_parquet("../data/training/anno-lexical-train.parquet", index=False)
dev_df.to_parquet("../data/training/anno-lexical-dev.parquet", index=False)
test_df.to_parquet("../data/training/anno-lexical-test.parquet", index=False)

train_df.to_csv("../data/training/anno-lexical-train.csv", index=False)
dev_df.to_csv("../data/training/anno-lexical-dev.csv", index=False)
test_df.to_csv("../data/training/anno-lexical-test.csv", index=False)


# %%
print("Label distributions")
print(train_df.label.value_counts(normalize=True))
print(dev_df.label.value_counts(normalize=True))
print(test_df.label.value_counts(normalize=True))

print("Party distributions")
print(train_df.source_party.value_counts(normalize=True))
print(dev_df.source_party.value_counts(normalize=True))
print(test_df.source_party.value_counts(normalize=True))
# %%
print("Per party label distribution")

print("\nLeft")
print(
    train_df[train_df["source_party"] == "Left"].label.value_counts(
        normalize=True
    )
)
print(
    dev_df[dev_df["source_party"] == "Left"].label.value_counts(normalize=True)
)
print(
    test_df[test_df["source_party"] == "Left"].label.value_counts(
        normalize=True
    )
)

print("\nRight")
print(
    train_df[train_df["source_party"] == "Right"].label.value_counts(
        normalize=True
    )
)
print(
    dev_df[dev_df["source_party"] == "Right"].label.value_counts(
        normalize=True
    )
)
print(
    test_df[test_df["source_party"] == "Right"].label.value_counts(
        normalize=True
    )
)

# %%
print("Per source label distribution")

print("\nLeft")
print(
    train_df[train_df["source_name"] == "alternet"].label.value_counts(
        normalize=True
    )
)
print(
    dev_df[dev_df["source_name"] == "alternet"].label.value_counts(
        normalize=True
    )
)
print(
    test_df[test_df["source_name"] == "alternet"].label.value_counts(
        normalize=True
    )
)
# %%
