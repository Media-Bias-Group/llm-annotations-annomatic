import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re


def plot_distributions(df):
    sns.set(style="whitegrid")
    _, axes = plt.subplots(2, 1, figsize=(8, 6))
    sns.countplot(x="bias_rating", hue="media_bias", data=df, ax=axes[0])
    axes[0].set_title(
        "Distribution of Media Bias in Partisanship of the article"
    )
    axes[0].set_ylabel("Count")
    top_topics = df["topic"].value_counts().head(5).index
    df_filtered = df[df["topic"].isin(top_topics)]
    sns.countplot(x="topic", hue="media_bias", data=df_filtered, ax=axes[1])
    axes[1].set_title("Distribution of Media Bias in Topic")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("./media_bias_distribution.png")
    plt.show()

import pandas as pd
import functools

def to_parquet(filename):
    def decorator_to_parquet(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Call the original function
            result = func(*args, **kwargs)
            result.to_parquet(filename)
            
            return result
        return wrapper
    return decorator_to_parquet


def unify_site_name(site_name):
    """
    Unifies the site name by removing specifications in brackets, converting to lowercase,
    removing '.com', removing special characters, and replacing spaces with hyphens.

    Args:
        site_name (str): The site name to be cleaned.

    Returns:
        str: The cleaned site name.
    """
    # Remove the specification in brackets
    base_list = [
        "News",
        "Online News",
        "Online",
    ]  # List of specifications that refer to general news content
    specification = re.search(r"\((.*?)\)", site_name)

    # Remove the specification if it is in the base list
    if specification is not None:
        if specification.group(1) in base_list:
            site_name = re.sub(r"\((.*?)\)", "", site_name).rstrip(" ")

    site_name = site_name.lower()
    site_name = re.sub(".com", "", site_name)
    site_name = re.sub(r"[^\w\s]", "", site_name)
    site_name = site_name.replace(" ", "-")
    return site_name