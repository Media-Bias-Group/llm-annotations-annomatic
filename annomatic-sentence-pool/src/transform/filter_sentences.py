import re
import uuid

import pandas as pd
from langdetect import detect
from nltk.tokenize import sent_tokenize
from src.utils_ import to_parquet
from tqdm import tqdm

INPUT_PATH = "data/transform/tmp"
OUTPUT_PATH = "data/transform/tmp"
tqdm.pandas()


def _starts_with_lowercase(sentence):
    # Strip leading whitespaces
    trimmed_sentence = sentence.strip()
    if not trimmed_sentence:
        return True

    # Check if the sentence is non-empty and starts with a lowercase letter
    return trimmed_sentence[0].islower()


def _unify_text(text):
    """Unify text e.g. remove URLs, lowercase, etc.

    Use this method whenever we want to 'unify' text.
    :param rm_hashtag: Flag if remove hashtags from text.
    :param text: A plain text as string.
    :return: The unified text as string.
    """

    if text != text:
        return text

    text = re.sub(r"#[A-Za-z0-9_]+", "", text)  # remove #hashtag
    text = re.sub(r"RT\ ", " ", text)  # remove 'RT' from tweets
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)  # remove @user
    text = re.sub(r"https?://[A-Za-z0-9./]+", " ", text)  # remove links
    text = re.sub("\t", " ", text)  # remove tab
    text = re.sub("\n", " ", text)  # remove newlines
    text = re.sub("\r", " ", text)  # remove \r type newlines
    text = re.sub(r" +", " ", text)  # remove multiple whitespaces
    text = re.sub(r"linebreak", "", text)  # remove linebreaks
    return text


def _ends_regularly(sentence):
    # Strip trailing whitespaces
    trimmed_sentence = sentence.rstrip()

    # Define a list of regular ending characters including various quotation marks
    regular_endings = [
        ".",
        "?",
        "!",
        '"',
        "'",  # Straight quotes
        "\u201C",
        "\u201D",  # Curly double quotes
        "\u2018",
        "\u2019",  # Curly single quotes
        "\u00AB",
        "\u00BB",  # Double angle quotes
        "\u2039",
        "\u203A",  # Single angle quotes
        "\u201E",
        "\u201F",  # German quotes
        "\u00AB",
        "\u00BB",  # French Guillemets
        "\u300C",
        "\u300D",
        "\u300E",
        "\u300F",  # CJK quotes
    ]

    # Check if the sentence ends with any of the regular ending characters
    return trimmed_sentence and trimmed_sentence[-1] in regular_endings


def _contains_quotation(sentence):
    # Regular expression pattern for matching text within quotation marks
    pattern = r"\".*?\"|'.*?'|[“”]"
    # Search for the pattern in the sentence
    return re.search(pattern, sentence) is not None


def _split_into_sentences(article_text, title):
    """
    Splits an article text into sentences.

    Args:
        article_text (str): The article text to be split.

    Returns:
        list: The list of sentences.
    """
    sentences = [title] + sent_tokenize(article_text)

    def valid_sentence(sent):
        if sent is None:
            return False
        if len(sent) <= 10:
            return False
        return True

    sentences = [
        sentence for sentence in sentences if valid_sentence(sentence)
    ]
    return sentences


@to_parquet(f"{OUTPUT_PATH}/sentences.parquet")
def main():
    df = pd.read_parquet(f"{INPUT_PATH}/sentences.parquet")

    df = df[df.sentence != ""]
    df = df[~df.sentence.isna()]
    df["approx_len"] = df["sentence"].apply(lambda x: len(x.split(" ")))
    df = df[~df.sentence.duplicated()]
    df = df[df.approx_len > 10]
    df["sentence"] = df["sentence"].progress_apply(_unify_text)
    df = df[~df.sentence.apply(_starts_with_lowercase)]
    df = df[df.sentence.apply(_ends_regularly)]
    df = df[~df.sentence.apply(_contains_quotation)]

    return df


if __name__ == "__main__":
    main()
